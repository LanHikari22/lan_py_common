- [Column Descriptions in tasks.csv](#column-descriptions-in-taskscsv)
  - [TaskId](#taskid)
  - [OptIssue](#optissue)
  - [OptCommit](#optcommit)
  - [Owner](#owner)
  - [Status](#status)
  - [Created](#created)
  - [LastUpdated](#lastupdated)
  - [OptEntry](#optentry)
  - [Category](#category)
  - [Description](#description)

# Column Descriptions in tasks.csv

## TaskId
This field gets a unique task id.

## OptIssue
Optional field (can be left empty). Refers to an Issue number in remote.

## OptCommit
Refers to the latest relevant commit for code change relevant to the task.

## Owner
Who is doing the task

## Status
Status can be:
    - WIP: Work in Progress
    - FIN: Finished and pushed.
    - NEW: Registered but not currently active.
    - ACH: Archived.

## Created
Date when the task row was created

## LastUpdated
Date when the task was last touched or updated

## OptEntry
Optionally, there is an entries folder that can have additional notes on the task. It is of the form 
{TaskId}.*. for example a 5.md could explain further details on Task 5.

## Category
A rough short code to categorize the task

## Description
Descrition of the task. Be concise and actionable.