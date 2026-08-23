import { GithubRepoLoader } from "@langchain/community/document_loaders/web/github"

import {Document} from '@langchain/core/documents'
import { generateEmbedding, summariseCode } from "./gemini";
import { db } from "~/server/db";
import { Octokit } from "octokit";

const getFileCount = async(path:string,octoKit:Octokit,githubOwner:string,githubRepo:string, acc: number = 0)=>{
    
        const {data} = await octoKit.rest.repos.getContent({owner:githubOwner, repo: githubRepo,path})
    
        if(!Array.isArray(data) && data.type === 'file'){
            return acc + 1
        }
    
        if(Array.isArray(data)){
            let fileCount = 0;
            const directories: string[] = []
    
            for(const item of data){
                if(item.type === 'dir'){
                    directories.push(item.path)
                } else{
                    fileCount++;
                }
            }
    
            if(directories.length > 0){

                const directoryCounts = await Promise.all(
                    directories.map((dirPath) => getFileCount(dirPath, octoKit, githubOwner, githubRepo,0))
                )
    
                fileCount += directoryCounts.reduce((acc,count)=> acc + count,0)
            }
    
            return acc + fileCount
        }
        return acc

    


}

export const checkCredits = async(githubUrl : string, githubToken?: string)=>{
    const octoKit = new Octokit({auth:githubToken})

    const githubOwner = githubUrl.split('/')[3]
    const githubRepo = githubUrl.split('/')[4]

    if(!githubOwner || !githubRepo) return 0;

    try {
        // Validate URL structure
        const parsedUrl = new URL(githubUrl)
        const pathParts = parsedUrl.pathname.split('/').filter(Boolean)
        
        if (pathParts.length < 2) {
          throw new Error('Invalid GitHub URL format')
        }
        
        const [githubOwner, githubRepo] = pathParts
        if(!githubOwner || !githubRepo){
          throw new Error('Invalid GitHub URL format')
        }
        
        return await getFileCount('', octoKit, githubOwner, githubRepo)
      } catch (error) {
        throw new Error(
          error instanceof Error 
            ? `GitHub analysis failed: ${error.message}`
            : 'Unknown repository error'
        )
      }
    
}

export const loadGithubRepo = async(githubUrl : string, githubToken? : string)=>{

    const loader  = new GithubRepoLoader(githubUrl, {
        accessToken: githubToken || '',
        branch: 'main',
        ignoreFiles: ['package.json', 'yarn.lock', 'pnpm-lock.yaml'],
        recursive: true,
        unknown: 'warn',
        maxConcurrency: 5
    })

    const docs = await loader.load();
    return docs;
}

export const indexGithubRepo = async(projectId:string,githubUrl:string, githubToken?:string )=>{
    const docs = await loadGithubRepo(githubUrl,githubToken);
    const allEmbeddings = await generateEmbeddings(docs);

    await Promise.allSettled(allEmbeddings.map(async(embedding,index)=>{
        if(!embedding) return;

        const sourceCodeEmbedding = await db.sourceCodeEmbedding.create({
            data:{
                summary:embedding.summary,
                sourceCode: embedding.sourceCode,
                fileName: embedding.fileName,
                projectId:projectId
            }
        })

        await db.$executeRaw`
        UPDATE "SourceCodeEmbedding"
        SET "summaryEmbedding" = ${embedding.embedding} :: vector
        WHERE "id" = ${sourceCodeEmbedding.id}`
    }))
}

const generateEmbeddings = async(docs:Document[])=>{
    return Promise.all(docs.map(async (doc)=>{
        
        const summary = await summariseCode(doc)

        if(!summary){
            console.error("skipping file with no summary: ", doc.metadata.source)
            return null
        }

        const embedding  = await generateEmbedding(summary)
        return {
            summary,
            embedding,
            sourceCode: JSON.parse(JSON.stringify(doc.pageContent)),
            fileName: doc.metadata.source,
        }
    }))
}