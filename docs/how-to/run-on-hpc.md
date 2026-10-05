# Run fits on a computing cluster

When you have many participants, or a model that is slow to fit, you can spread the fits over the nodes of a computing cluster.
cpm fits participants independently of each other, so the simplest approach is to split the participants into chunks, and fit each chunk in a separate job.

This guide uses a [SLURM](https://slurm.schedmd.com/) job array, which runs the same script many times, once per chunk.
The same idea works with other schedulers.

## 1. A script that fits one chunk

Each task of a job array gets its own number in the environment variable `SLURM_ARRAY_TASK_ID`, and the number of tasks in `SLURM_ARRAY_TASK_COUNT`.
The script splits the participants into as many chunks as there are tasks, fits the chunk of its own task, and saves the results in a file named after the task.
Within a task, it fits participants in parallel on the cores that SLURM gave it (`SLURM_CPUS_PER_TASK`), see {doc}`parallelise-fitting`.

```{literalinclude} _code/fit_chunk.py
:language: python
:caption: fit_chunk.py
```

Replace the data and the model with your own.
You can test the script on your own computer: without the SLURM variables, it runs as a single task on one core.

## 2. A job script

The job script asks for resources for each task, and runs the fitting script.
`--array=0-9` creates 10 tasks, numbered 0 to 9.

```{literalinclude} _code/job.sbatch
:language: bash
:caption: job.sbatch
```

How you load Python and cpm depends on your cluster: ask your administrators, or see its documentation.
Create the `logs` folder before you submit the job, then submit it with:

```bash
mkdir -p logs
sbatch job.sbatch
```

`squeue --me` shows the state of your tasks.

## 3. Combine the results

When all tasks have finished, the `results` folder holds one file per task.
Combine them into one file:

```{literalinclude} _code/combine.py
:language: python
:caption: combine.py
```

## Choosing the size of the chunks

- The number of tasks times the number of cores per task is the number of participants fitted at the same time. There is no benefit in more tasks than participants.
- Each task starts Python and loads the data, which takes a few seconds; make the chunks large enough that fitting takes much longer than that.
- If a task fails, only its chunk has to be fitted again: resubmit that task alone with `sbatch --array=3 job.sbatch`.
- For hierarchical estimation ({py:mod}`cpm.hierarchical`), all participants must be fitted together in every iteration, so do not split them into chunks. Run the whole analysis as a single task, with many cores.
