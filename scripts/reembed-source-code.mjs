/**
 * One-off backfill: re-embed every SourceCodeEmbedding row.
 *
 * Why this is needed: rows were embedded with text-embedding-004, which Google shut
 * down on 2026-01-14. Its replacement embeds into a different vector space, so the
 * stored vectors are not comparable with newly generated query vectors — retrieval
 * silently returns nonsense until every row is regenerated.
 *
 * Rows whose summary is an empty string were indexed while gemini-1.5-flash was
 * already retired (summariseCode swallowed the 404 and returned ''). Those get a
 * fresh summary from the stored sourceCode first.
 *
 * Usage:
 *   DATABASE_URL=... GEMINI_API_KEY=... node scripts/reembed-source-code.mjs [--dry-run] [--project <id>]
 */

import { PrismaClient } from "@prisma/client";

const CHAT_MODEL = "gemini-3.7-flash";
const EMBEDDING_MODEL = "gemini-embedding-2";
const EMBEDDING_DIMENSIONS = 768;
const CONCURRENCY = 4;
const MAX_RETRIES = 5;
const API = "https://generativelanguage.googleapis.com/v1beta";

const args = process.argv.slice(2);
const dryRun = args.includes("--dry-run");
const projectId = args.includes("--project")
  ? args[args.indexOf("--project") + 1]
  : undefined;

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("GEMINI_API_KEY is not set");
  process.exit(1);
}

const db = new PrismaClient();

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function callGemini(path, body) {
  let lastError;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const res = await fetch(`${API}/${path}?key=${apiKey}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (res.ok) return res.json();

    const text = await res.text();
    lastError = new Error(`${res.status} ${text.slice(0, 200)}`);

    // Retry rate limits and transient server errors with exponential backoff.
    if (res.status === 429 || res.status >= 500) {
      const wait = 2 ** attempt * 1000;
      console.warn(`  ${res.status}, retrying in ${wait}ms`);
      await sleep(wait);
      continue;
    }
    throw lastError;
  }
  throw lastError;
}

async function summarise(sourceCode, fileName) {
  const res = await callGemini(`models/${CHAT_MODEL}:generateContent`, {
    contents: [
      {
        role: "user",
        parts: [
          {
            text: `You are an intelligent senior software engineer who specialises in onboarding junior engineers onto projects.
You are onboarding a junior software engineer and explaining to them the purpose of the ${fileName} file

Here is the code:
-----------

${sourceCode.slice(0, 10000)}

-----------
Give a summary no more than 100 words of the code above`,
          },
        ],
      },
    ],
  });
  return res.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
}

async function embed(text) {
  const res = await callGemini(`models/${EMBEDDING_MODEL}:embedContent`, {
    content: { role: "user", parts: [{ text }] },
    outputDimensionality: EMBEDDING_DIMENSIONS,
  });
  const values = res.embedding?.values;
  if (!Array.isArray(values) || values.length !== EMBEDDING_DIMENSIONS) {
    throw new Error(`expected ${EMBEDDING_DIMENSIONS} dims, got ${values?.length}`);
  }
  return values;
}

const stats = { done: 0, resummarised: 0, failed: 0, skipped: 0 };

async function processRow(row) {
  try {
    let summary = row.summary;

    if (!summary || !summary.trim()) {
      if (!row.sourceCode || !row.sourceCode.trim()) {
        console.warn(`skip ${row.fileName}: no summary and no sourceCode`);
        stats.skipped++;
        return;
      }
      summary = await summarise(row.sourceCode, row.fileName);
      if (!summary.trim()) {
        console.warn(`skip ${row.fileName}: re-summarise returned empty`);
        stats.skipped++;
        return;
      }
      stats.resummarised++;
    }

    const vector = await embed(summary);

    if (dryRun) {
      console.log(`[dry-run] ${row.fileName} -> ${vector.length} dims`);
      stats.done++;
      return;
    }

    const literal = `[${vector.join(",")}]`;
    await db.$executeRaw`
      UPDATE "SourceCodeEmbedding"
      SET "summary" = ${summary},
          "summaryEmbedding" = ${literal}::vector
      WHERE "id" = ${row.id}`;

    stats.done++;
    if (stats.done % 25 === 0) console.log(`  ...${stats.done} rows updated`);
  } catch (e) {
    stats.failed++;
    console.error(`FAILED ${row.fileName}:`, e.message);
  }
}

async function main() {
  const rows = await db.sourceCodeEmbedding.findMany({
    where: projectId ? { projectId } : undefined,
    select: { id: true, fileName: true, summary: true, sourceCode: true },
  });

  console.log(
    `re-embedding ${rows.length} rows with ${EMBEDDING_MODEL} @ ${EMBEDDING_DIMENSIONS} dims` +
      (projectId ? ` (project ${projectId})` : "") +
      (dryRun ? " [DRY RUN]" : ""),
  );

  const queue = [...rows];
  const workers = Array.from({ length: CONCURRENCY }, async () => {
    while (queue.length) {
      const row = queue.shift();
      if (row) await processRow(row);
    }
  });
  await Promise.all(workers);

  console.log(
    `\ndone: ${stats.done} updated, ${stats.resummarised} re-summarised, ` +
      `${stats.skipped} skipped, ${stats.failed} failed`,
  );

  if (stats.failed > 0) process.exitCode = 1;
}

main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(() => db.$disconnect());
