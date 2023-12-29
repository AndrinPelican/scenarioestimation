# Scenario Estimation

Implementation of the algorithm from the paper Scenario Estimation.

## Structure

- monte_carlo_experiment
    - contain the monte carlo experiments to see how the estimation works for different settings
        - Peer effects
        - Network transitivity
        - Comparison simulation to exact solution (for reciprocity)

- estimation
    - Implementation of the estimators for Peer Effects and Networks
    - bringing together bucket sampling, model parameters and optimisation
    - *Sample Buckets* Contains the algorithms from the paper on how to sample buckets

- Application
    - using scenario estimation to understand transitivity in Nyakatoke
    - estimation of peer effects for drug use (no data so far)

- test
    - unit tests for the components

- util

## Estimation Types

- Network formation
    - reciprocity
    - transitivity

- Peer Effects
    - peer relation ship as a digraph
    - weighted peer relationship

## Peer effects:

- Working OK

- Has bias when geometric random graph
    - Bias diminishes when considering 100 buckets (needed in the case of N=1000, linkProb 0.75)
    - how bias reduction works depends weight distribution


- Weight distribution
    - The variance in the weights depends on the amount of circle in the network
        - circle free graph: no weight variance, independent of ordering of draw shock order
        - Erdos Reny: small variance
        - 5 friends among 10 neighbours in linear order: medium variance
        - random geometric, with random order: medium variance, depends on the link probability in neighborhood
        - block matrix: height difference

    - It seems that the probability without gamma also can help to decrease weight distribution
        - see OtherFile

- Distribution Coverage:
    - tail problem, about 10 % redjections!
    - exact case works well (mut about 100 games)
    - seems not to be a problem with weight distribution
        - erdös reny reciprocal or not reciprocal are about the same
    - 100 games with each 20 player and 50 buckets work well, maybe buckets decisive

- calculating hessian with a separate estimation of the log likelyhood
    - sometimes giving NaN in standard error, not known why..
    - seems not to have a impact on the wald/likelyhood/stdError test,


- See if you should weight the different games with separate estimation (and buckets) gradient and weight

## Network case:

Working well with transitivity and reciprocity for the erdös reny case

Not working well when degrees are considered

- reciprocity upward bias (more buckets seem to help)
- transitivity downward bias (very strong, seems to become better when bigger network)
- If the other parameters are fixed to the true parameters during the optimisation it works!

It seems to be the problem of the MLE method, not our way to evaluate the ML function and score:

- For the reciprocity case, we can solve the MLE closed form
- The behaviour was similar as when evaluated with strategy estimation

Behaviour improves when considering bigger networks.

