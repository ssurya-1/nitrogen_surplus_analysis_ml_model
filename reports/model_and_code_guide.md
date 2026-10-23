# What the model does
The model estimates three things for a crop in a European region in a given year: the crop yield (t/ha), the nitrogen removed in the harvest (kg N/ha), and the nitrogen surplus (kg N/ha/yr). Surplus is the N left over after the harvest has taken its share, and it is the standard indicator of how much N could leak into water or air.

These three outputs are linked by the soil-surface N budget. Everything that goes onto the field (mineral fertiliser, manure, deposition from the air, fixation by soil bacteria, and N returned in crop residues) either leaves with the harvest or is counted as surplus:

surplus = mineral + manure + deposition + fixation + residues − removal          (1)

The model predicts yield and removal, then calculates surplus from Equation 1. Because surplus is calculated rather than guessed, the three outputs always add up. A model that predicted surplus on its own could report a surplus that does not match its own yield and fertiliser figures, and nobody checking the numbers would trust it.

The data covers soft wheat, barley, grain maize and rapeseed in {regions} NUTS2 regions (roughly, provinces or states) of {countries} EU countries and the UK, 2000 to 2018, giving {rows:,} rows. Each row is one crop in one region in one year.

# How the model is built
The model is a multilayer perceptron, a type of neural network. It takes a list of numbers (the inputs), multiplies them by learned weights, applies a simple non-linear function (ReLU, which sets negative values to zero), and repeats. Training adjusts the weights so the outputs match the known answers.

- One network serves all four crops. A crop embedding, four learned numbers per crop, tells the network which crop a row is, so the crops can share what they learn while still behaving differently.
- The shared part has two hidden layers of 128 and 64 units. Dropout switches off 10% of units at random during training, which stops the network memorising the training rows.
- Each crop then has its own small output layer that gives two numbers: yield and removal.
- A final fixed step calculates surplus from Equation 1. This step has no weights to learn; it is the budget written into the model.

Training compares all three outputs with the real values. The surplus error is measured after the budget step, so when surplus is wrong the correction flows back into the removal output too. That is why the outputs are learned together in one model rather than by separate models for each output.

# How the code is organised
| File | What it does |
|---|---|
| data_prep.py | Reads the raw CAPRI and HaNi-Crop files and builds the per-crop budget table |
| models.py | Defines the network, the budget step, the loss and the training loop |
| evaluate.py | Splits the data, trains the model, scores it and builds the hotspot ranking |
| predict.py | Gives one prediction for a chosen region, crop and year |
| figures.py | Draws every figure, in report and slide sizes |
| report_tools.py | Shared Word formatting for the documents |
| build_results_document.py | Builds the results document and the key-number list |
| build_final_report.py | Builds the final report from reports/report_text.md |
| build_guide.py | Builds this guide from reports/model_and_code_guide.md |
| scripts/style_check.py | Checks every file for em dashes, stock AI phrases, comment-heavy code and unfilled placeholders |
| nitrogen_mtl.ipynb | Walks through the pipeline and shows the main outputs |

## Data preparation (data_prep.py)
The script has three steps, run in order.

### Step 1: extract
CAPRI stores every variable as a 1 km map for each year, about 1,700 maps in total. Each map cell carries the value of its Farm Structure Unit (a small farming area), so averaging all cells in a region gives the regional value. load_regions reads the map of unit IDs and links each unit to its NUTS2 region. zonal_sum then adds up any map by region in one fast step (numpy's bincount). extract_budget averages the N-budget maps weighted by farmland area, and extract_crops adds up crop areas and averages yields weighted by crop area. The results are cached in data/interim/ so this slow step (about five minutes) runs once.

### Step 2: hani
HaNi-Crop gives fertiliser and manure rates per crop on a coarser global grid (0.1°, about 10 km). cell_lonlat converts each 1 km CAPRI cell to latitude and longitude, and hani_rates looks up the HaNi value at that point and averages it over each region, weighted by the crop's area in each cell. region_geometry also records each region's centre point and which regions share a border, which the spatial test needs.

### Step 3: budget
build_budget assembles the per-crop budget:

- Removal. calibrate_n_content finds how much N one tonne of each crop removes, by fitting CAPRI's regional N removal against area times yield of all 35 crops (non-negative least squares, R² = {nnls_r2:.2f}). Wheat ({n.SWHE.used:.1f} kg N/t) and barley ({n.BARL.used:.1f} kg N/t) came out in the literature range; maize and rapeseed did not and use literature values ({n.MAIZ.used:.0f} and {n.RAPE.used:.1f} kg N/t).
- Fertiliser and manure. HaNi's raw rates jump in 2017, one of the test years. The code therefore takes each region-crop's average HaNi level over 2000-2014 and scales it year by year with CAPRI's own regional fertiliser series. The CAPRI ratio also converts HaNi's gross rates to CAPRI's soil-surface basis (N after surface losses), so the budget uses one definition throughout.
- Deposition, fixation and residues come from CAPRI's regional values (free_living_fixation estimates the part of fixation that applies to non-legume crops).

write_audit saves the checks the report relies on in results/: coverage by crop and year, the budget identity check, the N contents, the HaNi against CAPRI comparison, and the 2017 jump statistics.

## The network (models.py)
FEATURES lists the inputs: region latitude and longitude, year, mineral, manure and deposition N, the crop's share of farmland, farmland area (log) and livestock density. TARGETS lists the three outputs.

Network is the model itself. It looks up the crop embedding, joins it to the inputs, passes them through the shared layers, and then picks the yield-and-removal output layer that belongs to the row's crop. Scaler rescales inputs and outputs to mean 0 and standard deviation 1 using training rows only, which helps training and keeps test data out of the scaling.

The budget step is a few lines in outputs:

```
removal = z[:, 1] * scaler.y_std[1] + scaler.y_mean[1]
surplus = b["n_in"] + (b["rho"] - 1) * removal
```

Here n_in is the known inputs (mineral, manure, deposition and fixation) and rho is the region's residue ratio, so (rho - 1) * removal adds residues and subtracts removal in one term. budget_residual measures how far any prediction is from Equation 1; for this model it is always zero apart from rounding, and the evaluation records it as a check.

train is the training loop. It uses the Adam optimiser, which adjusts each weight by a step sized from that weight's recent gradients. After every pass through the data (an epoch) it measures the error on the validation years, keeps a copy of the best weights, and stops when 25 epochs pass without improvement. This is early stopping, which prevents the network fitting noise in the training years. fit wraps it and returns a predict function.

## Training and testing (evaluate.py)
make_splits builds the two tests the brief asks for. The temporal split trains on 2000-2014, validates on 2015-2016 and tests on 2017-2018, so the model is judged on years it never saw. The spatial split holds out whole countries in five groups. Regions that share a border with a held-out region are also removed from training, because neighbouring regions have almost identical data and would let the model cheat. Assertions in the function stop the run if any training region touches a test region.

past_average_reference gives a yardstick that is not a model: it predicts each test row's yield as its region-crop's average over the training years (the crop's overall average where the region has no history). It is saved to results/reference_past_average.csv so the report can say how the model compares.

tune tries two learning rates and two weight-decay values on the validation years and keeps the best ({lr:g} and {weight_decay:g}). run_split then trains the model with five random seeds on each split and saves the predictions. error_metrics computes MAE (the average size of the error), RMSE (like MAE but large errors count more), bias (the average of predicted minus actual, which shows whether the model leans high or low) and R² (the share of variation explained), plus the largest budget residual. hotspots ranks every 2017-2018 crop-region-year by predicted surplus per tonne of yield and checks how stable the ranking is between the two years.

## Figures and documents
figures.py draws each figure twice through export, once for the report (6.5 inches wide, 300 dpi) and once for slides (16:9, larger text). The colours come from a palette checked for colour-blind readers.

build_results_document.py reads results/, writes reports/key_numbers.csv (every headline value with a name such as temporal.surplus.mae), and builds the results document. build_final_report.py reads reports/report_text.md, where numbers are written as named placeholders inside braces, and fills each one from key_numbers.csv. If the model is retrained, rebuilding the documents updates every number, so the text cannot drift from the results.

# Running it
One-off setup:

```
cd ~/Downloads/ENGG2112_ml_model
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Full run from raw data (about 10 minutes in total):

```
.venv/bin/python data_prep.py extract
.venv/bin/python data_prep.py hani
.venv/bin/python data_prep.py budget
.venv/bin/python evaluate.py
.venv/bin/python figures.py
.venv/bin/python build_results_document.py
.venv/bin/python build_final_report.py
.venv/bin/python build_guide.py
.venv/bin/python scripts/style_check.py
```

To look at results without rerunning anything, open nitrogen_mtl.ipynb in VS Code and choose Run All.

## Asking the model for one prediction
predict.py loads the saved model and prints its prediction next to the budget values:

```
.venv/bin/python predict.py --list DE
.venv/bin/python predict.py --region DE11 --crop wheat --year 2018
```

The second command gives:

```
DE110000 (Germany), soft wheat, 2018: test year (not seen in training)
Inputs: mineral 154, manure 60, deposition 13, fixation 4.5 kg N/ha; crop area 89,025 ha

                              Budget value         Model         Error
Yield (t/ha)                          6.00          7.97         +1.97
Removal (kg N/ha)                   120.41        158.91        +38.50
Surplus (kg N/ha)                   153.99        131.27        -22.73
Budget residual (kg N/ha)                           0.00
```

This example shows two findings at once. The model over-predicts yield by almost 2 t/ha in 2018, the drought year, because it has no weather input. And even when it is wrong, its outputs still add up: the residual is zero, so the surplus error comes straight from the removal error and can be traced. Use real region-years only; the tool is for explaining the model, not for fertiliser advice.

## Changing things safely
- Test years: FIT_LAST, VAL_YEARS and TEST_YEARS in evaluate.py. TRAIN_YEARS in data_prep.py must end at the same year as FIT_LAST, because the HaNi level is averaged over the training years.
- Seeds: SEEDS in evaluate.py.
- Adding a crop: add it to HANI and CROP_NAMES in data_prep.py and to CROPS in models.py, then rerun from the budget step. The crop needs a HaNi-Crop file.
- Report wording: edit reports/report_text.md and rebuild; never type numbers into the Word file by hand.
- Run the project's own scripts without Python's -I flag, otherwise they cannot import each other.
- A rerun may change the last decimal place of some results, because training uses random starting weights. The five-seed average keeps this small.

# Questions you may be asked
### Why a neural network and not XGBoost, as in the proposal?
Two reasons. The budget step sits inside the model, and the network is trained on the surplus error measured after that step, so surplus mistakes also correct the removal output. Tree models such as XGBoost fit each output separately and cannot be trained through a formula like that. A network also lets the four crops share one model through the crop embedding.

### Why only one model?
The assignment asks for one model, and its marking criteria do not require a comparison. The team's brief originally planned a comparison with unconstrained models; the report's plan-versus-delivered table records that change and the reason.

### How do you know the budget rule works?
Every prediction is checked against Equation 1. The largest residual across all test predictions was {residual_max_all:.5f} kg N/ha, which is rounding error.

### Why is surplus R² so high (above 0.9)?
Most of the surplus is set by the inputs, which the model is given. Once the inputs are known, surplus follows from removal through Equation 1. The high R² mostly reflects that algebra, so the report treats yield and removal errors as the real test.

### Is calculating surplus from the budget cheating?
No. It does not use the answer. It uses only the known inputs and the model's own predicted removal, and it guarantees the outputs are physically consistent. The report claims exact conservation only for our constructed budget, not for CAPRI's unmeasured losses.

### How do you know there is no data leakage?
Anything calculated from the targets (current yield, residues, removal and CAPRI's surplus) is excluded from the inputs. Scaling uses training rows only. The test years come after the training years, and in the spatial test whole countries are held out with a border buffer, checked by assertions in the code.

### Why region coordinates instead of a region or country label?
A label has no learned meaning for a country the network never saw, so it would break the held-out-country test. Coordinates carry over to new places.

### Why are errors higher for held-out countries?
In a new country the model has seen no data from the region at all, so it must judge how productive the land is from location and inputs alone. Yield error is {spatial_rise:.0%} higher than on later years. It captures differences between regions well for wheat (R² {spatial.SWHE.yield_r2:.2f}) and poorly for maize ({spatial.MAIZ.yield_r2:.2f}).

### Does the model use past yields as an input?
No. An earlier version was given each region's average yield from the training years. It was more accurate on later years but less accurate in new countries, where that input could only be filled with a crop-wide average, so it was removed (see the development log). Reusing a region's own past average has an error of {ref.temporal.yield_mae:.2f} t/ha on later years against {temporal.yield_tha.mae:.2f} for the model, which is why the report presents it as a reference point. The model's value is that it works in regions with no records and returns removal and surplus that balance.

# Glossary
| Term | Meaning |
|---|---|
| CAPRI | An EU agricultural model whose disaggregated dataset gives crop areas, yields and N budgets for every region, 2000-2018 |
| HaNi-Crop | A global dataset of fertiliser and manure N rates by crop (Hao and Tian), used with the authors' permission |
| NUTS2 | The EU's standard second-level regions, roughly provinces or states |
| FSU | Farm Structure Unit, the small mapping unit CAPRI's maps are built from |
| UAAR | Utilised agricultural area, all farmland in a region |
| Soil-surface budget | The N balance at the soil surface: inputs minus harvested removal equals surplus |
| Hard constraint | A rule built into the model's structure so it can never be broken; here, surplus calculated from the budget |
| Embedding | A short list of learned numbers that represents a category, here the crop |
| Epoch | One full pass of the training data through the network |
| Early stopping | Ending training when validation error stops improving |
| Seed | The random starting point for training; five seeds give an average and a spread |
| MAE, RMSE | Average error size; RMSE weights large errors more |
| Bias | Average of predicted minus actual; positive means over-prediction |
| Budget residual | How far predicted removal and surplus are from satisfying Equation 1 |
