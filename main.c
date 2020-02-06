#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <mpi.h>

#define MASTER 0

uint64_t arrayLength;
uint32_t *p_array;

void initArray()
{
    for(int i = 0; i<arrayLength; i++)
    {
        //create 32bit random number
        uint32_t x = rand() & 0xff;
        x |= (rand() & 0xff) << 8;
        x |= (rand() & 0xff) << 16;
        x |= (rand() & 0xff) << 24;
        p_array[i] = x;
    }
}

uint64_t computeSerial()
{
    uint64_t sum = 0;
    clock_t start = clock();
    for(int i = 0; i<arrayLength; i++)
    {
        sum += p_array[i];
        if(sum < p_array[i])
        {
            printf("Overflow in serial addition!");
            exit(1);
        }
    }
    clock_t end = clock();
    printf("Start time: %lu \n", start);
    printf("End time: %lu \n", end);
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Duration: %.2lf milliseconds\n", seconds*1000);

    printf("Serial sum: %" PRIu64 "\n", sum);
    return sum;
}

uint64_t update(int myoffset, int chunk, int myid)
{
    uint64_t mysum = 0;
    /* Perform addition to each of my array elements and keep my sum */
    for(int i=myoffset; i < myoffset + chunk; i++)
    {
        mysum = mysum + p_array[i];
        if(mysum < p_array[i])
        {
            printf("Overflow in parallel addition!");
            exit(1);
        }
    }
    return mysum;
}

int main(int argc, char *argv[])
{
    printf("Initializing array....\n");

    uint64_t parallelSum;
    uint64_t serialSum;
    srand(time(NULL));

    int numTasks, taskId;
    int rc, dest, offset, i, j, tag1, tag2, source, chunksize, leftover;
    uint64_t mysum, totalSum;
    MPI_Status status;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskId);

    arrayLength = strtoll(argv[1], NULL, 10);
    p_array = (uint32_t *)malloc(sizeof(uint32_t)*arrayLength);

    if(taskId == MASTER)
    {
        if(argc == 1)
        {
            printf("Please specify array length as cli argument!");
            exit(0);
        }
        else if(argc > 2)
        {
            printf("More than one cli argument specified!");
            exit(0);
        }

        if(p_array == NULL)
        {
            printf("malloc of size %" PRIu64 "failed!\n", arrayLength);
            exit(1);
        }

        initArray();

        serialSum = computeSerial();
    }

    chunksize = (arrayLength / numTasks);
    leftover = (arrayLength % numTasks);
    tag2 = 1;
    tag1 = 2;

    /***** Master task only ******/
    if (taskId == MASTER)
    {
        clock_t start = clock();
        /* Send each task its portion of the array - master keeps 1st part plus leftover elements */
        offset = chunksize + leftover;
        for (dest=1; dest<numTasks; dest++)
        {
            MPI_Send(&offset, 1, MPI_INT, dest, tag1, MPI_COMM_WORLD);
            MPI_Send(&p_array[offset], chunksize, MPI_UINT32_T, dest, tag2, MPI_COMM_WORLD);
            offset = offset + chunksize;
        }

        /* Master does its part of the work */
        offset = 0;
        mysum = update(offset, chunksize+leftover, taskId);

        /* Wait to receive results from each task */
        for (i=1; i<numTasks; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, tag1, MPI_COMM_WORLD, &status);
            MPI_Recv(&p_array[offset], chunksize, MPI_UINT32_T, source, tag2,
                     MPI_COMM_WORLD, &status);
        }

        /* Get final sum and print sample results */
        MPI_Reduce(&mysum, &totalSum, 1, MPI_UINT64_T, MPI_SUM, MASTER, MPI_COMM_WORLD);
        clock_t end = clock();
        printf("Start time: %lu \n", start);
        printf("End time: %lu \n", end);
        float seconds = (float)(end - start) / CLOCKS_PER_SEC;
        printf("Duration: %.2lf milliseconds\n", seconds*1000);

        printf("Parallel sum: %" PRIu64 "\n", totalSum);

        if(serialSum != totalSum)
        {
            printf("Different sums calculated!");
            exit(1);
        }
    }  /* end of master section */



    /***** Non-master tasks only *****/

    if (taskId > MASTER)
    {
        /* Receive my portion of array from the master task */
        source = MASTER;
        MPI_Recv(&offset, 1, MPI_INT, source, tag1, MPI_COMM_WORLD, &status);
        MPI_Recv(&p_array[offset], chunksize, MPI_UINT32_T, source, tag2,
                 MPI_COMM_WORLD, &status);

        /* Do my part of the work */
        mysum = update(offset, chunksize, taskId);

        /* Send my results back to the master task */
        dest = MASTER;
        MPI_Send(&offset, 1, MPI_INT, dest, tag1, MPI_COMM_WORLD);
        MPI_Send(&p_array[offset], chunksize, MPI_UINT32_T, MASTER, tag2, MPI_COMM_WORLD);

        /* Use sum reduction operation to obtain final sum */
        MPI_Reduce(&mysum, &totalSum, 1, MPI_UINT64_T, MPI_SUM, MASTER, MPI_COMM_WORLD);

    } /* end of non-master */

    MPI_Finalize();

    free(p_array);
    return 0;
}

