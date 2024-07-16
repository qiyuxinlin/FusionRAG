# Start with website

This document provides the necessary steps to set up and run the web service for this project.

## 1. Starting the Web Service

### 1.1. Pip Installation

First, you need to install the necessary Python packages via pip. In your terminal, run the following command:

```bash
pip install ktransformers
```

Then, to start the web service, execute the following command:

```bash
ktransformers --web True
```

Finally, the web service is up and running, you can access it through your web browser. Simply type the following URL into your browser's address bar:

```
http://localhost:9016/web/index.html#/chat
```

### 1.2. Compiling the Web Code

Before you can compile the web code, make sure you have npm installed.

Once npm is installed, navigate to the `ktransformers/website` directory:

```bash
cd ktransformers/website
```

Next, install the Vue CLI with the following command:

```bash
npm install @vue/cli
```

Now you can build the project:

```bash
npm run build
```

Finally, start the web service again:

```bash
ktransformers --web True
```
