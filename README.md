# Delta Hedging and Option Pricing Project

A self-contained toolkit for exploring option valuation and delta hedging. The repository currently includes a binomial pricing notebook and is designed to grow to include Black-Scholes analytics and dynamic hedging experiments.

## Project Goals
- Compare binomial and Black–Scholes models for European options.
- Build intuition for Greeks (delta, gamma, vega, theta, rho) and how they inform hedging.
- Prototype delta-hedging strategies on simulated price paths and measure hedge error/P&L.

## Repository Structure
- **Binomial_Pricing.ipynb** – Notebook implementing binomial tree pricers (e.g., Cox–Ross–Rubinstein and Jarrow–Rudd variants) for European options.
- **README.md** – Project overview, setup, and usage instructions.

Planned additions include a Black-Scholes analytics module and dedicated hedging notebooks to compare discrete rebalancing strategies.

## Requirements
The existing notebook relies on common scientific Python libraries:

- Python 3.9+
- NumPy
- SciPy
- pandas
- matplotlib

If you do not have these installed, you can create a minimal environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib
```

## Getting Started
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd MATH_467_Project
   ```
2. **Set up dependencies** using the commands above (or your preferred environment manager).
3. **Open the notebook**
   ```bash
   jupyter notebook Binomial_Pricing.ipynb
   ```
   Run through the cells to see tree construction, option pricing outputs, and basic visualizations.

## Usage Notes
- Update the option parameters (spot price, strike, volatility, risk-free rate, maturity) at the top of the notebook to explore different scenarios.
- Use the existing binomial trees as a baseline for later comparisons against Black–Scholes analytics and hedging simulations.
- Feel free to duplicate the notebook to prototype additional trees (e.g., Leisen–Reimer) or to experiment with American exercise features.

## Roadmap
- Add a Black–Scholes pricing module with closed-form prices and Greeks.
- Create a delta-hedging notebook to simulate discrete rebalancing on geometric Brownian motion paths.
- Add plotting utilities and summaries to compare hedge performance across assumptions (rebalance frequency, transaction costs, and volatility misspecification).

## Contributing
Pull requests and issues are welcome. Please include a brief description of changes and any relevant plots or results when proposing new hedging experiments.

## License
Specify a license (e.g., MIT, BSD-3-Clause) if you plan to share or build on this work publicly.
