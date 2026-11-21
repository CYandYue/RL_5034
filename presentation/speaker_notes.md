# English presentation notes

Target: about 12 minutes including pointing at figures and transitions. Suggested team split: slides 1–4, 5–8, 9–12. Rehearse against the actual PDF; timing is a guide.

## 1. Question and scope (0:00–0:45)

Our project studies PPO hyperparameters for single-room HVAC control. The goal is to understand how clip range, GAE lambda, and update epochs affect comfort, electricity, and training stability. A teammate supplied the thermal simulator and physical validation. We added the training and evaluation pipeline, ran controlled experiments locally, and analyzed the results. We focus on this synthetic task and do not claim measured savings in a real building.

## 2. Physical environment (0:45–1:45)

The environment has two equivalent temperatures: the indoor node and the envelope node. Thermal resistances determine heat flow, while capacitances store energy and create inertia. The controller sees eleven numbers, including both temperatures, weather, gains, comfort bounds, time, and scenario. Its action controls heating or cooling power every fifteen minutes. The reward penalizes discomfort and electricity. We also report the physical components separately, because a better weighted reward may trade a little comfort for lower energy.

## 3. PPO mechanism (1:45–2:50)

PPO first collects a rollout with the current policy. The critic and GAE estimate whether sampled actions were better or worse than the policy's usual outcome in those states. The actor then changes their likelihood through the clipped objective. Lambda controls the advantage estimator. Clip range changes the surrogate objective. Epochs determine how often we reuse a rollout. The actor and critic have separate small networks. Gradients pass through action probabilities, not through the physical room equations.

## 4. Experimental protocol (2:50–3:50)

We varied one factor at a time around the common default. Sharing the default gives seven configurations, and three seeds each give twenty-one seeded runs. Every run receives exactly five hundred and twelve thousand environment interactions. Architecture, reward, learning rate, gamma, normalization, and sampling settings are fixed. Validation and final-test conditions are separate. We test each run's final checkpoint instead of choosing its best test result. The rules use the same physical conditions, and all experiments run on an M4 CPU under screen.

## 5. Learning curves (3:50–4:50)

This slide shows the validation cost, which is minus return, on a logarithmic scale. Lower is better. The large early penalties reflect poor temperature regulation. Later differences concern electricity and residual discomfort, as well as possible policy deterioration. The full report includes ordinary return plots and across-seed bands. A lower training loss is not a guarantee of monotonic closed-loop improvement, so the fixed final-checkpoint rule is important.

## 6. Final metrics (4:50–6:00)

The table separates return, comfort violation rate, and electricity. Default: 0.07 percent CVR, 13.99 kWh; thermostat: 0.01 percent, 14.94 kWh. Each PPO entry first averages its thirty test conditions, then averages across three training seeds. The error bars describe training variability; thirty episodes do not provide thirty independent trained models. HVAC off has zero electricity but can have severe discomfort. The thermostat is a meaningful conventional reference. We avoid calling one method universally better when comfort and electricity move in different directions.

## 7. Clip range (6:00–6:55)

Here we compare the three clip ranges while keeping lambda and epochs fixed. Mean return, small/default/large: -1.508, -1.417, -1.929. Clipping discourages additional gains from excessive probability-ratio changes in the preferred direction. It does not impose a hard bound on the final ratio or KL divergence. We therefore examine approximate KL, clipped fraction, learning curves, and final physical outcomes together. The ranking is specific to this experiment and comes from only three seeds.

## 8. GAE lambda (6:55–7:50)

Here gamma stays fixed while lambda changes how TD residuals are combined. Mean return, small/default/large: -1.501, -1.417, -1.377. Larger lambda uses longer reward fragments, which can reduce dependence on short-horizon value bootstrapping but increase variability. Thermal inertia makes delayed effects relevant, yet it does not imply that the largest lambda must win. The comparison also depends on the value network and rollout settings, so we describe the observed results without a universal monotonic claim.

## 9. Epochs and computation (7:50–8:50)

Epochs change how many times a rollout is optimized. At our common sample budget, five, ten, and twenty epochs correspond to forty thousand, eighty thousand, and one hundred and sixty thousand mini-batch updates. Mean training time, 5/10/20 epochs: 113/129/195 seconds. We compare sample efficiency at equal interactions and report wall time separately. More computation can improve data reuse but can also drive stronger policy changes. The runtime numbers are practical local measurements, affected by parallel load and machine temperature.

## 10. Closed-loop trace (8:50–9:50)

This is the preselected cold-weather test condition zero. The default PPO uses seed zero, and the thermostat starts from identical conditions. The comfort band provides a reference for indoor temperature. The lower plot shows cumulative electricity, not instantaneous power. Different locations within the comfort band can lead to different energy use. The report also shows outdoor forcing, executed actions, and the hot and transition scenes. This is an illustration; aggregate conclusions use every fixed test condition and training seed.

## 11. Stability and limitations (9:50–10:55)

We retain late validation deterioration because we test the final checkpoint. Largest mean late-validation peak-to-final drop: Default, 0.19 return units. The diagnostics are complementary rather than standalone failure criteria. Our small validation realization contains only negative offsets in hot and transition weather. Final tests include warmer offsets and reveal sensitivity in one large-clip seed; independently reloading that model reproduces its metrics exactly. Three seeds and one-factor comparisons also limit our conclusions. The simulator uses repeating weather, constant COP, and direct envelope-temperature observation, without real-building calibration.

## 12. Conclusion (10:55–12:00)

Highest observed mean PPO return: lambda=0.99 (-1.377); this is a descriptive grid result. Our main outcome is a reproducible account of how these parameters affect learning, comfort, energy, stability, and compute in one controlled task. We do not collapse all these dimensions into a single universal recommendation. The complete workflow runs on a local Mac and includes tests, fixed scenarios, models, logs, and generated figures. Useful extensions would add more seeds, study a limited set of interactions, or introduce calibrated dynamics and realistic weather. Thank you, and we welcome questions.
