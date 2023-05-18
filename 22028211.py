"""Analyzing Top 20 Countries' Electric power consumption (kWh): K-Means Clustering 
and Polynomial Models for Hungry and Saudi Arabia """


"""import libraries"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

"""import data set"""
df = pd.read_csv('Electric power consumption (kWh per capita).csv')

"""checking some basic information"""
df.head()

df['Country Name']

df.columns

df.isnull().sum()


"""Correlation Matrix"""

# Compute the correlation matrix
corr_matrix = df.iloc[:, 2:].corr()  # Assuming the numerical columns start from the third column

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.xlabel('Year', fontsize=10, fontweight='bold')
plt.ylabel('Year', fontsize=10, fontweight='bold')
plt.show()

"""Time Series Analysis"""

# Extract the relevant columns
years = df.columns[2:]  # Assuming the year columns start from the third column

# Plot the time series
plt.figure(figsize=(10, 6))
for index, row in df.iterrows():
    country = row['Country Name']
    power_consumption = row[2:]
    plt.plot(years, power_consumption, label=country)

plt.xlabel('Year')
plt.ylabel('Electric Power Consumption (kWh per capita)')
plt.title('Electric Power Consumption by Country (Time Series)')
plt.grid(True)
# Adjust the x-axis ticks and labels
# Adjust the legend position
plt.legend(loc='upper left', bbox_to_anchor=(0, 1.50), ncol=6)  # Move legend to the top

# Adjust the x-axis ticks and labels
plt.xticks(range(len(years)), years, rotation=90, ha='center')
plt.tick_params(axis='x', which='both', bottom=False, top=True)
plt.tight_layout()
plt.show()


"""Calculate the WCSS"""

# Extract the numerical columns for clustering
numerical_columns = df.columns[2:]  # Assuming the numerical columns start from the third column
X = df[numerical_columns].values
# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Calculate the WCSS for different numbers of clusters
wcss = []
max_clusters = 10  # Maximum number of clusters to consider

for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_normalized)
    wcss.append(kmeans.inertia_)  # Inertia represents the WCSS

# Plot the WCSS graph
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Within-Cluster Sum of Squares (WCSS)')
plt.show()


"""Step 3: Apply clustering to the dataset using KMeans algorithm and normalize the data

"""

# Apply K-means clustering:
k = 2  # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(X_normalized)

# Add the cluster labels as a new column in the dataset:
df['Cluster'] = kmeans.labels_

# Visualize the cluster membership and cluster centers:
plt.figure(figsize=(10, 6))
for cluster_label in range(k):
    cluster_df = df[df['Cluster'] == cluster_label]
    plt.scatter(cluster_df['Country Name'], [cluster_label] * len(cluster_df), label=f'Cluster {cluster_label + 1}')

plt.scatter(df['Country Name'], df['Cluster'], marker='.', color='black')
plt.scatter(kmeans.cluster_centers_[:, 0], range(k), color='red', marker='x', label='Cluster Centers')

plt.xlabel('Country')
plt.ylabel('Cluster')
plt.title('Cluster Membership and Cluster Centers')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

"""make a subset of country with subset"""

# Subset the data for Australia (AUS) and energy consumption columns
aus_energy_data = df.loc[df['Country Code'] == 'AUS', ['Country Code', 'Country Name', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014']]

# Display the subsetted data
aus_energy_data

"""Step 5: Create a simple model for fitting the data using curve_fit

"""

# Subset the data for Qatar (QAT) and energy consumption columns
qatar_energy_data = df.loc[df['Country Code'] == 'QAT', ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014']]

# Convert the year columns to numeric type
qatar_energy_data = qatar_energy_data.apply(pd.to_numeric, errors='coerce')

# Remove rows with missing values
qatar_energy_data = qatar_energy_data.dropna()

# Check if there are valid data points
if qatar_energy_data.empty:
    print("No valid data points for Qatar's energy consumption.")
else:
    # Create x values (years)
    x = np.arange(2000, 2015)

    # Fit a polynomial regression model to the energy consumption data
    degree = 2  # Set the degree of the polynomial
    coeffs = np.polyfit(x, qatar_energy_data.values.flatten(), degree)

    # Generate the corresponding y values based on the polynomial regression
    y = np.polyval(coeffs, x)

    # Generate x values for prediction
    prediction_years = np.arange(2015, 2020)  # Next five years

    # Extend the x array to include the prediction years
    x_pred = np.append(x, prediction_years)

    # Generate the corresponding y values for prediction
    y_pred = np.polyval(coeffs, x_pred)

    # Plot the polynomial regression line
    plt.figure(figsize=(10, 6))
    plt.plot(x, qatar_energy_data.values.flatten(), 'o', label='Actual Data')
    plt.plot(x_pred, y_pred, label='Prediction')
    plt.plot(x, y, label='Polynomial Regression')
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption')
    plt.title('Polynomial Regression for Qatar')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display predicted energy consumption for the next five years
    print("Predicted Energy Consumption for the Next Five Years:")
    for year, consumption in zip(prediction_years, y_pred[-5:]):
        print(f"Year: {year}, Consumption: {consumption}")

"""err_range

"""

# the error range refers to the difference between the actual data points and the predicted values
# Define the error range function
def err_ranges(model, xdata, conf=0.95):
    alpha = 1.0 - conf
    y_pred = model.predict(xdata)
    n = len(xdata)
    p = model.coef_.shape[0]
    dof = max(0, n - p - 1)
    tval = scipy.stats.t.ppf(1.0 - alpha / 2., dof)
    se = np.sqrt(np.sum((y - y_pred) ** 2) / dof)
    conf_range = tval * se
    lower = y_pred - conf_range
    upper = y_pred + conf_range
    return lower, upper

from sklearn.linear_model import LinearRegression
import scipy.stats


# Check if there are valid data points
if qatar_energy_data.empty:
    print("No valid data points for qatar's energy consumption.")
else:
    # Concatenate the data into a single time series
    x = np.arange(2000, 2015).reshape(-1, 1)
    y = qatar_energy_data.values.flatten()

    # Fit polynomial regression model
    model = LinearRegression()
    model.fit(x, y)

    # Generate predictions
    x_pred = np.arange(2000, 2015).reshape(-1, 1)
    y_pred = model.predict(x_pred)

    # Estimate the confidence range
    lower, upper = err_ranges(model, x_pred)

    # Plot the data, polynomial regression line, and confidence range
    # A confidence interval is a range of values that is likely to contain the true population parameter with a certain level of confidence
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data')
    plt.plot(x_pred, y_pred, 'r-', label='Polynomial Regression')
    plt.fill_between(x_pred.flatten(), lower, upper, color='gray', alpha=0.3, label='Confidence Range')
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption')
    plt.title('Energy Consumption in qatar (Polynomial Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Subset the data for Saudi Arabia (SAU) and energy consumption columns
sau_energy_data = df.loc[df['Country Code'] == 'SAU', ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014']]

# Convert the year columns to numeric type
sau_energy_data = sau_energy_data.apply(pd.to_numeric, errors='coerce')

# Remove rows with missing values
sau_energy_data = sau_energy_data.dropna()

# Check if there are valid data points
if sau_energy_data.empty:
    print("No valid data points for Saudi Arabia's energy consumption.")
else:
    # Create x values (years)
    x = np.arange(2000, 2015)

    # Fit a polynomial regression model to the energy consumption data
    degree = 2  # Set the degree of the polynomial
    coeffs = np.polyfit(x, sau_energy_data.values.flatten(), degree)

    # Generate the corresponding y values based on the polynomial regression
    y = np.polyval(coeffs, x)

    # Generate x values for prediction
    prediction_years = np.arange(2015, 2020)  # Next five years

    # Extend the x array to include the prediction years
    x_pred = np.append(x, prediction_years)

    # Generate the corresponding y values for prediction
    y_pred = np.polyval(coeffs, x_pred)

    # Plot the polynomial regression line
    plt.figure(figsize=(10, 6))
    plt.plot(x, sau_energy_data.values.flatten(), 'o', label='Actual Data')
    plt.plot(x_pred, y_pred, label='Prediction')
    plt.plot(x, y, label='Polynomial Regression')
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption')
    plt.title('Polynomial Regression for Saudi Arabia')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display predicted energy consumption for the next five years
    print("Predicted Energy Consumption for the Next Five Years:")
    for year, consumption in zip(prediction_years, y_pred[-5:]):
        print(f"Year: {year}, Consumption: {consumption}")

"""make error range"""

from sklearn.linear_model import LinearRegression
import scipy.stats
if sau_energy_data.empty:
    print("No valid data points for Saudi Arabia's energy consumption.")
else:
    # Concatenate the data into a single time series
    x = np.arange(2000, 2015).reshape(-1, 1)
    y = sau_energy_data.values.flatten()

    # Fit polynomial regression model
    model = LinearRegression()
    model.fit(x, y)

    # Generate predictions
    x_pred = np.arange(2000, 2015).reshape(-1, 1)
    y_pred = model.predict(x_pred)

    # Estimate the confidence range
    lower, upper = err_ranges(model, x_pred)

    # Plot the data, polynomial regression line, and confidence range
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data')
    plt.plot(x_pred, y_pred, 'r-', label='Polynomial Regression')
    plt.fill_between(x_pred.flatten(), lower, upper, color='gray', alpha=0.3, label='Confidence Range')
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption')
    plt.title('Energy Consumption in Saudi Arabia (Polynomial Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()


""" Compare Countries"""


num_cols = [col for col in df.columns if col != 'Country Name']
df['Average'] = df[num_cols].mean(axis=1)

# Create custom color palette
colors = sns.color_palette('husl', len(df))

# Create bar chart
sns.set(style='whitegrid')
plt.figure(figsize=(12, 8), dpi=300)
ax = sns.barplot(x='Average', y='Country Name', data=df, palette=colors)

# Add creativity - change the face color of bars
for i, patch in enumerate(ax.patches):
    color = colors[i % len(colors)]
    patch.set_facecolor(color)

# Customize labels, ticks, and aesthetics
ax.set(xlabel='Average Energy Consumption', ylabel=None)
ax.tick_params(axis='both', which='major', labelsize=12, width=2, pad=5, length=5)
plt.yticks(rotation=45, ha='right')
plt.title('Average Energy Consumption by Country', fontsize=16, fontweight='bold')

# Add creativity - annotate bar values
for i, patch in enumerate(ax.patches):
    width, height = patch.get_width(), patch.get_height()
    x, y = patch.get_xy()
    ax.annotate(f'{width:.2f}', (x + width / 2, y + height / 2),
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Add creativity - add background pattern
ax.set_facecolor('#f5f5f5')
ax.grid(color='white', linestyle='-', linewidth=1, axis='y')

plt.show()