import { z } from "zod";
import { createTRPCRouter, protectedProcedures } from "../trpc";
import { pollCommits } from "~/lib/github";
import { indexGithubRepo } from "~/lib/github-loader";

export const projectRouter = createTRPCRouter({
    createProject: protectedProcedures.input(
        z.object({
            name: z.string(),
            githubUrl: z.string(),
            githubToken: z.string().optional()
        })
    ).mutation(async ({ ctx, input }) => {
        const project = await ctx.db.project.create({
            data: {
                githubUrl: input.githubUrl,
                name: input.name,
                userToProjects: {
                    create: {
                        userId: ctx.user.userId!,
                    }
                }
            }
        })
        
        console.log('indexing github repo');
        await indexGithubRepo(project.id,input.githubUrl,input.githubToken)
        await pollCommits(project.id)
        
        return project;
    }),

    getProjects: protectedProcedures.query(async ({ ctx }) => {
        return await ctx.db.project.findMany({
            where: {
                userToProjects: {
                    some: {
                        userId: ctx.user.userId!
                    },
                },
                deletedAt: null
            }
        })
    }),
    
    getCommits: protectedProcedures.input(
        z.object({
            projectId: z.string(),
        })
    ).query(async({ctx,input})=>{
        pollCommits(input.projectId).then().catch(console.error);
        
        return await ctx.db.commit.findMany({
            where: { 
                projectId : input.projectId
            }
        })
    })
})

