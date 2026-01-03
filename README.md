# 🤖 ML Playground

An interactive web app where you can explore machine learning hands-on — right in your browser. Built with React, TypeScript, and TensorFlow.js, it lets you train and visualize different models in real time without needing any backend or server setup.

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://iripvanwinkle.github.io/ml-playground/)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](#)

⭐ Like this project? Give it a star on GitHub — it really helps!

## Overview

ML Playground is built to make machine learning easier to understand through experimentation. You can train models, tweak their settings, and see how they learn — all in real time.

From simple linear regression to neural networks and decision trees, the app provides a clear window into how algorithms work under the hood.

> [!NOTE]  
> We use TensorFlow.js mainly for tensor operations, but most algorithms are implemented from scratch. This way, you can peek into the math and logic behind the models instead of treating them like black boxes.

## ✨ Features

- **Train interactively**: Watch models learn in real time with live progress updates
- **Experiment with algorithms**: Try linear/logistic regression, neural networks, and more
- **Visualize your data**: Explore results with interactive Plotly.js charts
- **Bring your own data**: Upload CSVs or use built-in sample datasets

## 🎮 Usage Guide

1. **Choose Your Task**  
   Select a machine learning task: **Regression** (predicting continuous values), **Classification** (categorizing data), or **Clustering** (grouping similar data points).

2. **Select or Upload Data**  
   Use a sample dataset or upload your own CSV. The app takes care of preprocessing and visualization.

3. **Configure Your Model**  
   Adjust algorithm-specific settings such as regularization, criteria, or distance metrics. Build neural networks layer by layer with custom activations, or configure clustering parameters and number of clusters.

4. **Set Training Parameters**  
   Choose your optimizer and learning strategy, set learning rate, batch size, and number of iterations. Enable regularization and configure convergence criteria as needed.

5. **Train & Watch**  
   Hit **Start Training** and see loss curves, accuracy metrics, and predictions update live. Try different setups to see what works best.

### Supported Algorithms

#### Regression

- **Linear Regression**: Simple and multiple linear regression with various optimizers
- **Neural Networks**: Multi-layer perceptrons with customizable architectures
- **Decision Trees**: Tree-based regressors with ensemble methods

#### Classification

- **Logistic Regression**: Binary and multi-class classification
- **Softmax Regression**: Multi-class classification with softmax output
- **One-vs-Rest**: Multi-class classification using binary classifiers
- **Neural Networks**: Deep learning for classification tasks
- **Decision Trees**: Tree-based classification with ensemble methods
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem

#### Clustering

- **K-Means**: Unsupervised clustering algorithm for grouping similar data points

## 🚀 Quick Start

### Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd ml-playground
    ```

2. Install dependencies:

    ```bash
    npm install
    ```

3. Start the development server:

    ```bash
    npm run dev
    ```

4. Open your browser and navigate to `http://localhost:5173`

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run test` - Run test suite
- `npm run test:watch` - Run tests in watch mode
- `npm run test:coverage` - Run tests with coverage report
- `npm run lint` - Check code quality
- `npm run format` - Format code with Prettier
- `npm run check` - Run all quality checks (lint, format, typecheck, test)

## 🏗️ Architecture

ML Playground follows a layered architecture with clear separation of concerns:

- **App Layer** (`src/app/`): React components, UI sections, and state management
- **ML Layer** (`src/ml/`): Pure ML algorithms independent of the UI framework
- **Web Workers**: Training operations run in background threads for performance

### Key Technologies

- **Frontend**: React 19, TypeScript, Tailwind CSS v4
- **ML Library**: TensorFlow.js + custom implementations
- **Visualization**: Plotly.js for interactive charts
- **State Management**: Zustand with actions pattern
- **UI Components**: Radix UI primitives with shadcn/ui
- **Build Tool**: Vite with ES modules support
- **Testing**: Vitest with comprehensive coverage

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`npm run check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow TypeScript best practices
- Write tests for new features
- Use conventional commit messages
- Ensure code passes all quality checks (`npm run check`)

## 📚 Learn More

### Machine Learning Concepts

- [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Neural Networks](https://en.wikipedia.org/wiki/Neural_network)
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)

### Technologies Used

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Vite Documentation](https://vitejs.dev/)

---

**Happy Learning! 🎉** Start exploring machine learning concepts interactively at [ml-playground](https://iripvanwinkle.github.io/ml-playground/)
