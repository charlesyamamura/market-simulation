import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as random
from jax import vmap

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.preprocessing import MinMaxScaler
# Imported regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Data Preparation ---
def load_and_scale_data(filepath):
    df = pd.read_excel(filepath)
    
    # Chronological split
    train_df = df[df['year'] <= 2018].copy()
    test_df  = df[df['year'] == 2019].copy()
    
    # Separate features and target
    drop_cols = ['year', 'mo', 'model', 'share']
    X_train_raw = train_df.drop(columns=drop_cols).values
    y_train_raw = train_df['share'].values.reshape(-1, 1)
    
    X_test_raw = test_df.drop(columns=drop_cols).values
    y_test_raw = test_df['share'].values.reshape(-1, 1)
    
    # Scale features and target to assist MCMC convergence
    scaler_X = MinMaxScaler().fit(X_train_raw)
    scaler_y = MinMaxScaler().fit(y_train_raw)
    
    X_train = scaler_X.transform(X_train_raw)
    y_train = scaler_y.transform(y_train_raw)
    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw)
    
    return (jnp.array(X_train), jnp.array(y_train.squeeze()), 
            jnp.array(X_test), jnp.array(y_test.squeeze()), scaler_y)

# --- 2. BNN Model Architecture ---
def nonlin(x):
    return jnp.tanh(x)

def bnn_model(X, Y=None, D_H=10):
    N, D_X = X.shape

    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros((D_H,)), jnp.ones((D_H,))))
    z1 = nonlin(jnp.matmul(X, w1) + b1)
    
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros((D_H,)), jnp.ones((D_H,))))
    z2 = nonlin(jnp.matmul(z1, w2) + b2)
    
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, 1)), jnp.ones((D_H, 1))))
    b3 = numpyro.sample("b3", dist.Normal(jnp.zeros((1,)), jnp.ones((1,))))
    
    z3 = (jnp.matmul(z2, w3) + b3)[..., 0]
    sigma = numpyro.sample("sigma", dist.Gamma(1.0, 1.0))
    
    with numpyro.plate("data", N):
        numpyro.sample("obs", dist.Normal(z3, sigma), obs=Y)

# --- 3. Main Execution and Inference ---
def main(args):
    X_train, y_train, X_test, y_test, scaler_y = load_and_scale_data(args.data_path)
    
    print("Starting MCMC Sampling on Metal Device...")
    nuts_kernel = NUTS(bnn_model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains
    )
    
    mcmc.run(random.PRNGKey(42), X_train, y_train, D_H=args.hidden_dim)
    mcmc.print_summary()
    
    print("\nGenerating predictions on test set (2019)...")
    posterior_samples = mcmc.get_samples()
    
    predictive = Predictive(bnn_model, posterior_samples=posterior_samples)
    predictions = predictive(random.PRNGKey(43), X_test, D_H=args.hidden_dim)["obs"]
    predictions_matrix = np.array(predictions)
    
    # Inverse transform predictions back to original market share percentages
    num_samples, num_records = predictions_matrix.shape
    predictions_rescaled = scaler_y.inverse_transform(predictions_matrix.reshape(-1, 1)).reshape(num_samples, num_records)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate summary statistics across the posterior distributions
    mean_prediction = np.mean(predictions_rescaled, axis=0)
    lower_bound = np.percentile(predictions_rescaled, 5.0, axis=0)
    upper_bound = np.percentile(predictions_rescaled, 95.0, axis=0)
    
    # --- 4. Performance Indicators Computation ---
    mae = mean_absolute_error(y_test_actual, mean_prediction)
    rmse = np.sqrt(mean_squared_error(y_test_actual, mean_prediction))
    r2 = r2_score(y_test_actual, mean_prediction)
    
    print("\n" + "="*40)
    print("      BNN TEST PERFORMANCE METRICS (2019)")
    print("="*40)
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared Score (R²):           {r2:.4f}")
    print("="*40)
    
    # Plotting actual vs predicted with Performance Metadata
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    index = np.arange(len(y_test_actual))
    
    ax.plot(index, y_test_actual, "ro-", label="Actual Market Share", alpha=0.8)
    ax.plot(index, mean_prediction, "b-", label="Mean BNN Prediction", lw=2)
    ax.fill_between(index, lower_bound, upper_bound, color="lightblue", alpha=0.5, label="90% Confidence Interval")
    
    ax.set_xlabel("Test Data Points (2019 Chronological)")
    ax.set_ylabel("Market Share")
    
    # Dynamically inject metrics into title for reporting clarity
    ax.set_title(f"BNN Market Share Forecast\nMAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
    ax.legend(loc="upper left")
    
    plt.savefig("bnn_market_share_eval.png")
    plt.show()
    print("Evaluation plot updated with performance indicators and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Neural Network for Market Share Analytics")
    parser.add_argument("--data_path", type=str, default="data1319.xlsx")
    parser.add_argument("-n", "--num-samples", default=1500, type=int)
    parser.add_argument("--num-warmup", default=500, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--hidden-dim", default=12, type=int)
    
    args, unknown = parser.parse_known_args()
    main(args)