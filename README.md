# Fair K-Core Algorithm

This repository contains the implementation of the Fair K-Core algorithm, which computes the maximum fair k-core in a graph with attributed nodes. The project is based on the research paper titled "Computing Fair K-Core in Attributed Graphs" by [Author(s)].

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

The Fair K-Core algorithm tackles the problem of identifying dense subgraphs in attributed graphs that satisfy fairness constraints based on the attributes of the nodes. The algorithm is designed to be efficient and scalable, with pruning optimizations and parallel programming techniques.

## Installation

To use the Fair K-Core algorithm, you'll need to have a C++ compiler installed on your system. The code has been tested with GCC and Clang compilers.

1. Clone the repository:

```bash
git clone https://github.com/your_username/fair_k_core.git
```

2. Change to the project directory:

```bash
cd fair_k_core
```

3. Compile the code:

```bash
g++ graph.cpp -o graph -std=c++20
```

This will generate an executable file named `fair_k_core`.

## Usage

To run the Fair K-Core algorithm on a graph, you'll need to provide the input graph in edge-list format as well as attribute file.  The format of the edgelist file should be as follows:

```
# <number_of_nodes> <number_of_edges>
<node_id_0> <node_id_1>
<node_id_1> <node_id_2>
...
```

To run the algorithm on an input graph, use the following command:

```bash
./graph <input_graph_file> <attribute_file> <k_value> <threshold>
```

where:
- `<k_value>` is the value of k for computing the k-core.
- `<threshold>` is the threshold for unfairness in the graph.

The output will be the maximum fair k-core of the graph, with information about the size of the fair k-core and connected components.

## License

This project is released under the [MIT License](LICENSE).