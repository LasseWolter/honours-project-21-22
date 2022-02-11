### Understanding of the problem

- justify design choices

  - how you explain what you have done is critical to 'Understanding the problem'
  - Leaving the reader confused seems like you are confused

- write dissertation directed to WHAT you did not your original goals
  - state why you changed these goals accordingly

### Completion of the project

- future work is important as it shows that you have thought of further steps
- concrete evidence that you have done something
- demonstrate why certain objectives weren't achieved
  -> don't focus on this though -> focus on what you did achieve
- introduction and conclusion
  - compare your objectives/goals with what you did

### Quality of work

- really asses what you have done
- make clear what the hard bits are
- ## better to do thorough testing with less features

### Quality of report

- structure
- logical journey
- no foolish subsections (1.1 but no 1.2) -> so it doesn't make sense to have that section
- poor figures -> make sure figures are proper

### Additional Criteria

- talk to supervisor about each of these points and if you have done them
- not just present prior work -> but interpret

  - show how your work fits into the picture
  - make this reference back in your discussion at the end

- is it efficient, secure

  - what are limitations of your solution
  - not just lots of results
    - interpretation is key -> make sure it's clear that you understand what these results mean
    - compare to certain benchmarks? compare to previous work?

- justify design decisions

  - explain why your decisions were reasonable even if they didn't improve the state of the art
  - explain why a standard method is a standard method (e.g. using deep learning instead of MFCCs)

- present the material that was unclear in a better way (e.g. the code by Gillick et al.)
- explain the problems you ran into with dataloading and explain why Gillick et al's solution doesn't seem sustainable to you

  - with more data it wouldn't work at some point (just loading everything into ram seems like bad practice)
  - explain that and compare that to your implemenation using Lhotse

- try to include all your work (at least mention it)
- state your achievements in the introduction and refer back to it
- also include paths you followed that didn't work (otherwise it won't count)
  - mention things that took significant amount of effort (for me e.g. the dataloading part of the training)
    - possibly write clear code that compares the two methods and shows the huge improvement of using lhotse
