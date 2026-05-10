# RO Fouling Optimizer

A physics-based tool for optimizing CIP cleaning frequency in reverse 
osmosis plants during the EPC design phase.

## Live Demo
👉 Coming soon

## The Problem

In EPC design, CIP cleaning frequency is typically set as a static 
criterion (e.g. 3 times/year) based on manufacturer recommendations 
or engineering experience, without considering the actual seasonal 
variability of the feed water at the specific site.

An oversized cleaning frequency increases OPEX unnecessarily. 
An undersized frequency accelerates membrane degradation and 
increases replacement costs.

## What This Tool Does

Given the geographic coordinates of a coastal desalination plant 
and the design operating parameters, this tool:

1. Retrieves historical seasonal water quality data from Copernicus
2. Simulates membrane fouling evolution using a Darcy-based physics model
3. Generates a synthetic dataset of fouling scenarios
4. Trains a machine learning model to predict optimal cleaning intervals
5. Quantifies the economic impact of adjusting the design criterion

## Who Is This For

Process engineers at EPC companies working on seawater desalination 
projects in conceptual or basic design phases who need to justify 
the CIP frequency adopted in their dimensioning.

## Project Status

🚧 Under construction

## Tech Stack

- Python 3.12
- numpy, scipy, matplotlib, pandas
- scikit-learn
- Copernicus Marine Service (CMEMS)

## Author

Álvaro Menéndez — Process Engineer, Water Sector  
[LinkedIn](https://www.linkedin.com/in/alvaro-mendoza-bonilla)  
[GitHub](https://github.com/alvmenbon)