# MSc Thesis: Retail Promotion and Shipment Planning Model

This repository contains the model code, data analysis scripts, processed input files, and selected output files used for my MSc thesis in Supply Chain Management at Rotterdam School of Management, Erasmus University.

The thesis develops and analyses a stochastic optimization model for promotion planning, shipment planning, and inventory decisions in a retail supply chain context. The model studies how demand uncertainty, promotional demand effects, and product complementarity influence promotion timing, shipment capacity usage, inventory decisions, and lost sales.

## Project overview

Retail supply chains often face volatile demand, frequent promotions, and interdependencies between products. Promotions may increase demand for the promoted product itself, but may also affect demand for complementary products. At the same time, retailers must make operational decisions about shipment capacity, inventory, and service levels.

This project uses a two-stage stochastic programming approach to study these decisions. First-stage decisions include promotion planning and regular truck reservations. Second-stage decisions are scenario-dependent and include shipment quantities, sales, inventory, and lost sales.

The repository includes:

- AMPL model files used for the optimization model. (main branch)
- Processed data input files used by the models. (main branch)
- Python scripts used for the empirical data analysis and preparation of model inputs. (main branch)
- Selected model output files used to support the results presented in the thesis. (output files branch)
