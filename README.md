# Open Shop Scheduling Problem

## Problem Description

In the Open Shop Scheduling Problem (OSSP), a set of jobs needs to be processed on each machine in a shop. Each job consists of a sequence of tasks (called activities), where each task corresponds to processing the job on one of the machines. The constraints are as follows:
- Each job has one activity per machine.
- No two activities of the same job can be processed simultaneously.
- Each machine can process only one activity at a time.

The objective is to find a job sequence that minimizes the **makespan** — the total time required to process all jobs.

## Data

The instances used in this problem are from Taillard's benchmark. Each instance contains:
- **Number of jobs**: The total number of jobs to be scheduled.
- **Number of machines**: The total number of machines in the shop.
- **Processing times**: Each job has a specific processing time on each machine.
- **Machine assignments**: The order in which each machine processes the jobs.
- **Upper and lower bounds**: Provides the upper and lower bound makespan values for the instance.



