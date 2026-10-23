@summary
Excess nitrogen on farmland drives nitrate pollution of water and nitrous oxide emissions. Models that estimate crop yield and nitrogen surplus can show where fertiliser use should be checked, but when each output is predicted separately the results need not add up to the nitrogen that was applied, which undermines trust in them. This project built one neural network for four crops that predicts yield and harvested nitrogen removal and calculates nitrogen surplus from the soil-surface nitrogen budget, so its outputs always balance.

We built a per-crop budget for soft wheat, barley, grain maize and rapeseed in {regions} EU and UK regions from 2000 to 2018 ({rows:,} crop-region-years), using the CAPRI disaggregated dataset and the HaNi-Crop fertiliser dataset. The model was tested on later years (2017-2018) and on whole countries held out of training.

On later years the model predicted yield to within {temporal.yield_tha.mae:.2f} t/ha on average (R² = {temporal.yield_tha.r2:.2f}) and surplus to within {temporal.surplus.mae:.1f} kg N/ha, and every prediction satisfied the budget. Yield error was {spatial_rise:.0%} higher in countries the model had never seen. The largest error came from the 2018 drought, which the model could not see without weather inputs. Ranking crop-region-years by predicted surplus per tonne showed that {top_decile_area:.0%} of crop area produced {top_decile_surplus:.0%} of predicted surplus. We recommend adding weather data and directing field verification to the regions that stayed high-yield and high-surplus in both test years.

@body
# Introduction
## Problem and motivation
Nitrogen (N) is the nutrient that limits most crop yields, and synthetic fertiliser has made modern cereal production possible. The N that crops do not take up is lost as nitrate to groundwater and rivers, where it drives eutrophication and algal blooms, or as nitrous oxide, a greenhouse gas (Sutton et al., 2011; US EPA, 2025). The European Union has regulated this through the Nitrates Directive since 1991, yet N surplus remains high in many intensive farming regions (Batool et al., 2022).

The standard indicator is the soil-surface N budget (Figure 1). Inputs from fertiliser, manure, atmospheric deposition, biological fixation and crop residues are balanced against the N removed in the harvest. The remainder is the N surplus, in kg N per hectare per year. Surplus is an indicator of potential loss, not a measurement of leaching or emissions (Eurostat, 2013).

[[figure budget_schematic | The soil-surface nitrogen budget used in this project. Surplus is what remains when harvested removal is subtracted from all inputs.]]

Our proposal planned a fertiliser recommender for Australian field trials. During development the team adopted a project brief that asks a more basic question first. If a model predicts yield, removal and surplus as separate outputs, nothing forces them to agree with the inputs. A tool that reports a surplus that does not match its own yield and fertiliser figures cannot be trusted by an adviser or a regulator. This project builds the budget into the model so that cannot happen, and measures how accurate the resulting model is.

## Objectives
The project had four measurable objectives:

- O1. Build a harmonised per-crop dataset of yield (t/ha), harvested N removal (kg N/ha) and soil-surface N surplus (kg N/ha/yr) for four crops across EU and UK regions, 2000-2018.
- O2. Train one multi-task network with the N budget built in, and evaluate it with mean absolute error (MAE), root mean squared error (RMSE) and bias for each output, on later years and on unseen countries.
- O3. Verify that every prediction satisfies the budget, and find where and why the errors arise.
- O4. Rank crop-region-years by predicted surplus per tonne of yield, to show where N use could be checked first.

## Background and prior work
Machine learning has been used to predict both maize yield and nitrate loss from weather, soil and management data (Shahhosseini et al., 2019), but each output was modelled on its own. Multi-task learning trains one model on several related outputs so that what it learns for one helps the others (Caruana, 1997). Physics-informed machine learning goes further and builds known physical laws into the model or its training (Raissi et al., 2019). In crop modelling, DeepCGM applied mass conservation to crop-growth variables but left the nitrogen cycle for future work (Han et al., 2025). European N budgets themselves are well documented: Batool et al. (2022) reconstructed soil N surplus back to 1850, and the CAPRI model provides crop areas, yields and the full budget for every EU region (Koeble et al., 2026). To our knowledge no study has built a single cross-crop model with an N-budget constraint on a common European dataset. That is the gap this project addresses, although a narrow literature search cannot rule out an equivalent study.

# Data and preprocessing
## Sources
The primary source was the CAPRI disaggregated dataset (Koeble et al., 2026; CC BY 4.0). It provides 1 km rasters for 2000-2018 in which every cell carries the value of its Farm Structure Unit, of which there are 217,884. We used the crop area and yield rasters for all 35 crops and the 15 N-budget layers.

CAPRI reports its N budget per hectare of all farmland in a region, not per crop. Building a per-crop budget therefore needed crop-specific fertiliser and manure rates, which came from HaNi-Crop (Hao & Tian, 2026), a 0.1° global grid of annual N application by crop. HaNi-Crop is under peer review, and its authors ask to be contacted before results based on it are published. The team contacted them and received permission to use it in this project; the dataset itself is not redistributed in the project repository.

## Processing
Each raster was reduced to NUTS2 regions: crop areas were summed, yields were weighted by crop area, and budget terms by farmland area. CAPRI's own budget closed to within {identity_max:.3f} kg N/ha in every region-year, which confirmed the extraction. Crop-region-years with at least 1,000 ha of the crop were kept, giving {rows:,} rows from {regions} regions in {countries} countries.

The per-crop budget, in kg N per hectare of crop, was

surplus = mineral + manure + deposition + fixation + residues − removal          (1)

Removal is yield multiplied by grain N content. Grain N content was calibrated by regressing CAPRI's regional N retention on the area and yield of all 35 crops with non-negative least squares (R² = {nnls_r2:.2f}). Wheat ({n.SWHE.calibrated:.1f} kg N/t) and barley ({n.BARL.calibrated:.1f} kg N/t) fell inside literature ranges. Maize and rapeseed did not, so literature midpoints were used ({n.MAIZ.used:.0f} and {n.RAPE.used:.1f} kg N/t). Deposition was CAPRI's regional value, fixation was the free-living share estimated from the data ({fixation:.1f} kg N/ha), and residues used each region's ratio of residue N to retention.

HaNi-Crop rates were sampled at every 1 km cell and averaged per region, weighted by that crop's area. Gross application rates were converted to CAPRI's soil-surface basis with CAPRI's own regional loss ratios, so the budget kept a single definition. Two data problems were found and fixed. First, raw HaNi rates changed by more than 50% from the previous year in {jump_2017_raw:.0%} of 2017 rows, against about {jump_other_raw:.0%} in other years, a step that fell inside our test period. We therefore used each region's HaNi level from the training years only, scaled by CAPRI's annual fertiliser series, which reduced such jumps to {jump_2017_fixed:.1%}. Second, {rows_clipped:.0f} extreme rates above each crop's 99.5th percentile were capped. As an external check, HaNi's matched crops accounted for {hani_ratio:.0%} of CAPRI's mineral N, with a regional correlation of {hani_corr:.2f}. The gap was largest in grassland countries such as Ireland and the Netherlands, which HaNi does not cover.

## Inputs and leakage control
Model inputs were crop, region latitude and longitude, year, mineral and manure N per crop, deposition, the crop's share of farmland, farmland area and livestock density. Region coordinates were used instead of a region or country code, because a code has no meaning for a country held out of training. Anything derived from the targets (current yield, residues, removal or CAPRI's surplus) was excluded, and no input correlated with a target by more than {leak_max:.2f} (mineral N against surplus). An earlier version was also given each region's average yield from the training years. It was more accurate on later years but less accurate in unseen countries, where it could only be given a crop-wide average, so it was removed so that the model learns regional differences itself (Appendix B).

# Methods
## Model
The model is a multi-task neural network shared by all four crops (Figure 2). Each row's inputs are joined to a learned four-value crop embedding and passed through two hidden layers of 128 and 64 units (ReLU activation, 10% dropout). A small output layer for each crop then predicts two values, yield and removal. Surplus is not predicted. It is calculated from Equation 1 using the known inputs and the predicted removal, so the three outputs satisfy the budget by construction. This is a hard constraint: the residual left when the outputs are put back into Equation 1 is zero for every prediction.

Three choices shaped this design. A single shared network lets the four crops learn from each other's data, which matters for rapeseed and maize with fewer rows. The budget step sits inside the model, so the training loss on surplus is measured after it. That error therefore also corrects the removal output, which a set of separate models trained on each output could not do. Finally, a hard constraint was chosen over a training penalty because a penalty only discourages imbalance, while the hard version guarantees balance. The guarantee holds for our constructed budget, not for CAPRI's unobserved losses.

[[figure pipeline | Workflow from raw rasters to the evaluated model and the hotspot ranking.]]

## Training, validation and testing
Inputs and targets were standardised using training rows only. The loss was the mean squared error over the three standardised outputs. The network was trained with the Adam optimiser (Kingma & Ba, 2015), with batches of 256 and early stopping when the 2015-2016 validation error had not improved for 25 epochs. A small search on the validation years chose the learning rate ({lr:g}) and weight decay ({weight_decay:g}). The model was trained with five random seeds, and results are reported as the mean, with the spread across seeds as a measure of stability.

Two blocked tests followed the brief. The temporal test trained on 2000-2014, validated on 2015-2016 and tested on 2017-2018, so the model was judged on years after everything it learned from. The spatial test held out whole countries in five folds over 2000-2016. Any training region sharing a border with a test region was also removed, so that neighbouring regions with near-identical data could not leak into training; assertions in the code confirm the buffer. MAE and RMSE measure the typical and the large errors, and bias is the mean of predicted minus observed values, which shows whether the model leans high or low.

## From proposal to delivered system
The delivered project kept the proposal's aim and evaluation structure but changed the data, the model and the output. Table 1 summarises the main changes; the full log is in Appendix B.

Table: How the delivered work differs from the proposal and the team brief.
| Plan | Delivered | Reason |
|---|---|---|
| Australian GRDC and CSIRO field trials | CAPRI and HaNi-Crop, EU and UK regions | Team adopted the CAPRI brief; one consistent budget for 26 countries |
| XGBoost yield model and a minimum-N search | One multi-task network with the N budget built in | Research focus moved to outputs that obey the budget |
| Brief: compare independent, shared and constrained models | The constrained model only | The assignment asks for one model; the constraint is justified by the physics of the budget |
| Benchmark against GRDC N demand | Budget closure check and blocked tests | GRDC values do not apply to European regions |
| Fertiliser recommendation per scenario | Region ranking for verification | Regional data cannot justify farm advice |
| Split stratified by crop and conditions | Blocked tests by later years and held-out countries | Avoids leakage between neighbouring regions |
| HaNi rescaled to CAPRI totals (interim plan) | HaNi level with CAPRI annual dynamics | HaNi has no grassland and a 2017 step |

Data preparation took longer than the Gantt chart allowed because of the raster volume (8.8 GB) and the HaNi fixes [check: confirm against the team's actual timeline].

# Results and analysis
## Accuracy (objective O2)
Table 2 summarises the test results. On later years the model predicted yield with an MAE of {temporal.yield_tha.mae:.2f} t/ha and explained {temporal.yield_tha.r2:.0%} of the variation in yield. Removal followed yield closely (MAE {temporal.removal.mae:.1f} kg N/ha), and surplus was predicted to within {temporal.surplus.mae:.1f} kg N/ha. The spread across the five seeds was small (yield MAE ± {temporal.yield_tha.mae_sd:.2f} t/ha), so the result does not depend on a lucky start. Surplus R² was above 0.9 in both tests, but that figure mostly reflects the budget algebra: once inputs are known, surplus follows from removal. Yield and removal errors are the more informative measures.

[[table performance | Test results, mean of five training seeds. Bias is predicted minus observed.]]

On countries held out of training the yield MAE rose to {spatial.yield_tha.mae:.2f} t/ha and the surplus MAE to {spatial.surplus.mae:.1f} kg N/ha. In a new country the model has seen no data from the region, so it relies on location and inputs alone, and Figure 3 shows the effect differs by crop. Within each crop, held-out R² ranged from {spatial.MAIZ.yield_r2:.2f} for maize to {spatial.SWHE.yield_r2:.2f} for wheat, so the model captures differences between regions well for wheat but poorly for maize. On later years, wheat and barley were predicted best (R² {temporal.SWHE.yield_r2:.2f} and {temporal.BARL.yield_r2:.2f}), maize had the largest absolute error, and rapeseed the weakest R² ({temporal.RAPE.yield_r2:.2f}), probably because its yields vary little between regions compared with their swings from year to year.

As a reference point, reusing each region's own past average yield, which the model is not given, has an error of {ref.temporal.yield_mae:.2f} t/ha on later years, slightly better than the model. Where no history exists, as in held-out countries, that approach can only use a crop-wide average ({ref.spatial.yield_mae:.2f} t/ha against {spatial.yield_tha.mae:.2f} for the model). The model's value is that it works in regions without records and returns removal and surplus that balance, not that it beats a region's own past average where one exists.

[[figure accuracy_by_crop | Yield and surplus MAE by crop on later years and on held-out countries.]]

## Budget closure (objective O3)
Across all test predictions the largest budget residual was {residual_max_all:.5f} kg N/ha, the size of floating-point rounding, so every set of outputs balanced. This matters for use: a surplus that does not match the yield and fertiliser figures beside it would be rejected by any adviser checking the numbers, and an error of a few kg N/ha is enough to move a region up or down a ranking.

## Where the errors came from
The single largest error source was the 2018 drought. The model was close to unbiased in 2017 ({year.2017.yield_bias:+.2f} t/ha) but over-predicted every crop in 2018 ({year.2018.yield_bias:+.2f} t/ha). Among countries with at least 20 test rows, the 2018 over-prediction was largest in Sweden, Germany, Poland and Czechia (Figure 4). Romania, Bulgaria and Hungary, which had good harvests, were under-predicted. This matches the 2018 north-south "water seesaw" (Toreti et al., 2019), and the model could not detect it because it has no weather inputs. Luxembourg, with only six test rows, and Belgium, where HaNi covers little of CAPRI's fertiliser, had the largest surplus errors, followed by Austria and Poland (Appendix C).

[[figure bias_by_year | Mean yield error by crop and test year. Positive values are over-predictions.]]

## Hotspot ranking (objective O4)
The model ranked 2017-2018 crop-region-years by predicted surplus per tonne of yield. The ranking agreed closely with the budget values (Spearman correlation {spearman_min:.2f} to {spearman_max:.2f} by crop), and {same_quadrant_min:.0%} to {same_quadrant_max:.0%} of region-crops fell in the same quadrant in both years. The 10% of cases with the highest surplus per tonne covered {top_decile_area:.0%} of crop area but produced {top_decile_surplus:.0%} of predicted surplus. Two findings changed how the ranking should be read (Figure 5). The highest values came from very high inputs relative to yield: rapeseed in Latvia, Sweden and the United Kingdom, and barley in Spain. The lowest values were negative surpluses ({negative_surplus_share:.0%} of cases), where the crop removes more N than it receives and soil N is being drawn down. That is not efficient in any lasting sense.

[[figure hotspot_quadrant | Predicted yield against predicted surplus by crop, 2017-2018. Lines mark crop medians; the ten highest and ten lowest surplus-per-tonne cases are highlighted.]]

# End-to-end system
The model is one stage of a larger loop. Each year, new CAPRI or Eurostat statistics, HaNi-type fertiliser estimates and seasonal weather would feed the model, which produces a consistent yield, removal and surplus estimate for every crop and region, with the spread across seeds as an uncertainty band. These estimates go to a ranking and map dashboard for agri-environment agencies, river basin managers and the officials who design Common Agricultural Policy eco-schemes. The output is a short list of regions to examine, not a fertiliser rate. In those regions advisers would check with farm-level nutrient budgets and soil N tests, and where surplus is confirmed, offer targeted measures such as split application, catch crops or manure redistribution. Water-quality monitoring and repeat farm surveys then measure whether surplus fell, and those observations are fed back to retrain and recalibrate the model. A person makes every decision that affects a farm; the model's role is to decide where to look first.

# Recommendations, stakeholders, impact and ethics
## Recommendations
1. Keep the budget inside the model in any operational version. Outputs that always balance can be audited line by line, and the constraint cost nothing to apply at prediction time.
2. Add weather before any operational use. The 2018 drought caused the largest error in the test, and ERA5-Land growing-season temperature and rainfall (Muñoz-Sabater et al., 2021) are the obvious first inputs.
3. Direct field verification to the region-crops that stayed high-yield and high-surplus in both test years, concentrated in Germany, the United Kingdom, the Netherlands, Belgium and France. These combine high absolute surplus with yields that suggest inputs could be trimmed without large losses, which a field trial would need to confirm.
4. Report negative surpluses separately as soil N mining, not as success, so that a ranking does not reward practices that run down soil fertility.
5. Treat rankings for countries outside the training data with caution. The model separated regions within a crop well for wheat (R² {spatial.SWHE.yield_r2:.2f}) but poorly for maize ({spatial.MAIZ.yield_r2:.2f}), so rankings there should be confirmed against national data first.
6. Check HaNi inputs before relying on any single region. Latvian rapeseed topped the ranking with values that look like an input artefact.

## Stakeholders and impact
Farmers and their advisers are the people most affected, since any change in N use falls on their yields and costs. Agri-environment agencies and water authorities would be the direct users. Downstream communities, drinking-water utilities and coastal fisheries gain from lower nitrate loads, and lower surplus also cuts nitrous oxide emissions. Fertiliser suppliers have a commercial interest in the outcome, and data providers such as the CAPRI and HaNi teams carry responsibility for the inputs. The main benefit is efficiency of attention: a regulator with limited inspection capacity can start where {top_decile_surplus:.0%} of the surplus sits on {top_decile_area:.0%} of the land.

## Socio-technical issues at scale
- The model learns CAPRI's and HaNi's assumptions. A high score against them shows agreement with those models, not with field truth, and errors in a widely used source would spread into every decision built on it.
- Regional averages hide farm-level variation. Treating every farm in a high-surplus region as a polluter would be an ecological fallacy and unfair to farms that already manage N well.
- Published rankings can stigmatise regions and attract regulatory pressure before verification. Rankings should be shared with uncertainty bands and with the people affected before any public release.
- Data coverage is uneven. Countries with fewer records or poorer HaNi coverage get less accurate estimates, so their farmers face more risk of being misclassified.
- Surplus can be misread as measured pollution. It is a potential-loss indicator, and leaching depends on soil, rainfall and timing.
- Automation bias is a risk once a dashboard exists. Officials may defer to the ranking instead of checking it; the system design keeps a verification step for that reason.
- Cutting N in one region can move production elsewhere, so EU-wide surplus might not fall as much as regional figures suggest.
- Using a dataset under peer review carries an obligation to its authors. The team obtained their permission before use, does not redistribute the data, and a deployed system would need a formal licence once the dataset is published.

## Future work
The next steps are to add ERA5-Land weather and soil properties, to replace the constructed per-crop budget with farm survey data where it exists (for example the EU Farm Accountancy Data Network), to test the approach on more crops including grassland, and to run a small field validation in two or three of the persistent hotspot regions.

# Limitations and conclusion
The per-crop surplus is our construction, not an observation. Removal equals yield times a fixed grain N content, so yield and removal carry the same information, and surplus R² is inflated by the budget algebra. Deposition is a fixed average and cannot explain change over time, two crops relied on literature N contents, and in countries it has not seen the model separates regions within a crop only moderately (R² from {spatial.MAIZ.yield_r2:.2f} for maize to {spatial.SWHE.yield_r2:.2f} for wheat). On later years it is also slightly less accurate than reusing a region's own past average yield. The results are also specific to four arable crops at regional scale.

Within these limits the project met its objectives. A single multi-crop network with the nitrogen budget built in predicted yield on later years to within {temporal.yield_tha.mae:.2f} t/ha, kept every prediction consistent with the budget, and produced a stable ranking of where N surplus is concentrated. It should not set fertiliser rates until weather is added and the hotspots have been checked in the field.

@references
Batool, M., Sarrazin, F. J., Attinger, S., Basu, N. B., Van Meter, K., & Kumar, R. (2022). Long-term annual soil nitrogen surplus across Europe (1850-2019). Scientific Data, 9, 612. https://doi.org/10.1038/s41597-022-01693-9 [check: article number]
Caruana, R. (1997). Multitask learning. Machine Learning, 28(1), 41-75. https://doi.org/10.1023/A:1007379606734
Eurostat. (2013). Nutrient budgets: Methodology and handbook (Version 1.02). Eurostat and OECD.
Han, [check: authors] (2025). DeepCGM: [check: full title, journal and DOI].
Hao, W., & Tian, H. [check: full author list on the Zenodo record] (2026). HaNi-Crop: A harmonized and gridded global dataset of crop-specific nitrogen inputs from nitrogen fertilizer and manure [Data set, under review]. Zenodo. https://zenodo.org/records/21364978
Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations. https://arxiv.org/abs/1412.6980
Koeble, R., [check: co-authors] (2026). [check: title of the CAPRI data descriptor]. Scientific Data. https://doi.org/10.1038/s41597-026-06919-8
Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A., Albergel, C., Arduini, G., Balsamo, G., Boussetta, S., Choulga, M., Harrigan, S., Hersbach, H., Martens, B., Miralles, D. G., Piles, M., Rodríguez-Fernández, N. J., Zsoter, E., Buontempo, C., & Thépaut, J.-N. (2021). ERA5-Land: A state-of-the-art global reanalysis dataset for land applications. Earth System Science Data, 13(9), 4349-4383. https://doi.org/10.5194/essd-13-4349-2021
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045
Shahhosseini, M., Martinez-Feria, R. A., Hu, G., & Archontoulis, S. V. (2019). Maize yield and nitrate loss prediction with machine learning algorithms. Environmental Research Letters, 14(12), 124026. https://doi.org/10.1088/1748-9326/ab5268
Sutton, M. A., Howard, C. M., Erisman, J. W., Billen, G., Bleeker, A., Grennfelt, P., van Grinsven, H., & Grizzetti, B. (Eds.). (2011). The European nitrogen assessment: Sources, effects and policy perspectives. Cambridge University Press.
Toreti, A., Belward, A., Perez-Dominguez, I., Naumann, G., Luterbacher, J., Cronie, O., Seguini, L., Manfron, G., Lopez-Lozano, R., Baruth, B., van den Berg, M., Dentener, F., Ceglar, A., Chatzopoulos, T., & Zampieri, M. [check: author list] (2019). The exceptional 2018 European water seesaw calls for action on adaptation. Earth's Future, 7(6), 652-663. https://doi.org/10.1029/2019EF001170
US EPA. (2025, January 10). Agriculture sector emissions. United States Environmental Protection Agency. https://www.epa.gov/ghgemissions/agriculture-sector-emissions
