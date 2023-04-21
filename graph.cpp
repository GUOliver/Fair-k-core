#include <iostream>
#include <cstdio>
#include <ostream>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <iterator>
#include <vector>
#include <queue>
#include <stack>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <climits> 

#include <functional>
#include <limits>
#include <algorithm>

typedef struct {
	int n; // number of nodes
	int e; // number of edges

    std::vector<std::vector<int>> adj_list;
    std::vector<int> exist; // 判断节点是否被删除
    std::vector<int> degree;
    std::vector<int> k_core;
    // group value of each vertex
    std::vector<int> group;
    // vertices of each group
    // std::vector<std::vector<int>> group_vs;
    // which group is being selected (initialized with 0 for each group)
    std::vector<int> group_sel; 
    // 删除节点的时候会附带删除的节点
    std::vector<std::vector<int>> del_cost;

    int attr_dimension; // number of attributes in the graph
    // attribute value of each vertex
    int max_count_attribute;
    int min_count_attribute;
    std::vector<int> attribute;
    // vertices count of each vertex
    std::unordered_map<int, int> attribute_counter;
} graph;

void InsertSort(std::vector<int> &a);
int BinarySearch(std::vector<int> &a, int value);
int BinarySearch1(std::vector<int> &a, int value);
std::vector<int> getDifference(std::vector<int> A, std::vector<int> B);
std::vector<int> getUnion(std::vector<int> A, std::vector<int> B);

graph* read_edgelist(const char* filename);
void read_attribute(graph* g, char* attr_file);
void compute_k_core(graph* g, int k);
std::vector<int> compute_delCost(graph* g, int k, int node);
void group_pruning(graph* g, int k, int threshold);
int compute_unfairness(graph* g, std::unordered_map<int, int>& attribute_counter);
int compute_unfairness_for_group(graph* g, std::vector<int>& group_vertices);
void compute_attribute_counter(graph* g);
void FKC(graph* g, int threshold, int k);
void delete_node(graph* g, int node, bool update_attribute);
// void delete_nodes(graph* g, int node, bool update_attribute);
// void delete_nodes_cascade(graph* g, int node, bool update_attribute);
void print_graph(graph* g);
int get_k_core_size(graph* g); 
void print_k_core(graph* g);
void print_k_core_size(graph* g);
void get_connected_components(graph* g, std::vector<std::vector<int>>& components);
graph* get_subgraph(graph* g, const std::vector<int>& component);

std::ostream& operator<<(std::ostream& os, const std::vector<int>& v) {
    os << "[ ";
    for (int i = 0; i < v.size(); i++) {
        os << v[i] << " ";
    }
    os << "]";
    return os;
}

bool is_k_core(graph* g, int k) {
    for (int i = 0; i < g->n; i++) {
        if (g->exist[i] == 1 && g->degree[i] < k) {
            return false;
        }
    }
    return true;
}


graph* read_edgelist(const char* filename) {
    std::ifstream fs(filename);
    graph* g = new graph;

    std::string hash_mark;
    int n, e;
    fs >> hash_mark >> n >> e;  // # num_vertices num_edges
    g->n = n;
    g->e = e;

    g->exist.resize(n, 1);
    g->degree.resize(n, 0);
    g->k_core.resize(n, 1);
    g->group.resize(n, -1);

    g->adj_list.resize(n);
    g->del_cost.resize(n);

    int from, to;
    for (int i = 0; i < e; ++i) {
        fs >> from >> to;
        if (from >= n || to >= n) {
            std::cerr << "Error: vertex index out of bounds." << std::endl;
            delete g;
            return NULL;
        }
        g->degree[from]++;
        g->degree[to]++;
        g->adj_list[from].push_back(to);
        g->adj_list[to].push_back(from);
    }

    printf("- Total vertices: %d \n", g->n);
    printf("- Total edges: %d \n", g->e);
    return g;
}


// read attributes of vertices from file
void read_attribute(graph* g, char* attr_file) {
    g->attribute.resize(g->n);
    FILE* f = fopen(attr_file, "r");
    if (f == NULL) {
        printf("Cannot open the attribute file ! \n");
        exit(1);
    }

    int curr_node = 0;
    int curr_attr = 0;
    g->attr_dimension = 0;
    while (fscanf(f, "%d %d", &curr_node, &curr_attr) == 2) {
        if (curr_node >=  g->n) {
            printf("Erroneous attribute file with vertice %d\n", curr_node);
            exit(1);
        } 
        g->attribute[curr_node] = curr_attr;
        if (curr_attr > g->attr_dimension) {
            g->attr_dimension = curr_attr;
        }
    }

    if (curr_node != g->n - 1) {
        printf("Some vertices do not have attributes, Check the attribute file!\n");
        exit(1);
    }

    g->attr_dimension++;
    fclose(f);
    printf("Attribute dimension: %d, Finished reading attribute file\n", g->attr_dimension);

    return;
}


void compute_k_core(graph* g, int k) {
    std::queue<int> q;
    for (int i = 0; i < g->n; i++) {
        if (g->degree[i] < k) {
            q.push(i);
            g->exist[i] = 0;
            g->k_core[i] = 0;
        }
    }

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int i = 0; i < g->adj_list[u].size(); i++) {
            int v = g->adj_list[u][i];
            if (g->exist[v] == 1) {
                if (--g->degree[v] < k) {
                    q.push(v);
                    g->exist[v] = 0;
                    g->k_core[v] = 0;
                }
            }
        }
    }
}

// Fill in the vertices counters for each attribute after k-core computing
void compute_attribute_counter(graph* g) {
    for (int i = 0; i < g->n; ++i) {
        if (g->exist[i] == 1) {
            int attr = g->attribute[i];
            if (g->attribute_counter.find(attr) == g->attribute_counter.end()) {
                g->attribute_counter[attr] = 1;
            } else {
                g->attribute_counter[attr]++;
            }
        }
    }
}


// Computes the deletion cost of a node with the given id based on k-core value.
// Time Complexity: O(n^2)
std::vector<int> compute_delCost(graph* g, int k, int id) {
	std::vector<int> delCost_vertices;
    //因为这个计算delCost不能破坏原本的值，所以就用了copy
	std::vector<int> degree_copy = g->degree;
    std::vector<int> exist_copy = g->exist;

	std::queue<int> del;

	del.push(id);
	delCost_vertices.push_back(id); // minimum deletion cost is 1
	exist_copy[id] = 0;

	while (!del.empty()) {
		int q = del.front();
		del.pop();
		for (int i = 0; i < g->adj_list[q].size(); ++i)//遍历节点q的所有邻居
		{
            int nid = g->adj_list[q][i];//nid是这个邻居的名称
			if (exist_copy[nid] == 1 && --degree_copy[nid] < k) {
                del.push(nid);
                delCost_vertices.push_back(nid);
                exist_copy[nid] = 0;
			}
		}
	}
	// sort(delCost_id.begin(), delCost_id.end());
	return delCost_vertices;
}

// Construct groups for nodes in the k-core
// For each group g, |V(g)| is a lower bound of dc(v)
void group_pruning(graph* g, int k, int threshold) {
	int group_id = -1;
    // loop through all the vertices
	for (int i = 0; i < g->group.size(); ++i) {
        std::queue<int> queue = {};
        // existed after k-core pruning, degree = k, and has not been assigned a group
		if (g->exist[i] == 1 && g->degree[i] == k && g->group[i] == -1) {
			group_id++;
            queue.push(i);
            g->group_sel.push_back(0);
            // g->group_vs.push_back(std::vector<int>());
            g->group[i] = group_id;
            // g->group_vs[group_id].push_back(i);
		}

        // Core step to identify each core-group
		while (!queue.empty()) {
			int id = queue.front();
			queue.pop();
			for (int j = 0; j < g->adj_list[id].size(); j++)
            {
                int nid = g->adj_list[id][j];
                if (g->exist[nid] == 1 && g->degree[nid] == k && g->group[nid] == -1)
                {
                    queue.push(nid);
                    g->group[nid] = group_id;
                    // g->group_vs[group_id].push_back(nid);
                }
			}
		}
	}
}

// Compute the diff between the attribute with the largest vertices count and the one with the smallest vertices count
int compute_unfairness(graph* g, std::unordered_map<int, int>& attribute_counter) {
    // Find the attribute with the max/min vertices counts
    int max_attribute = 0;
    int min_attribute = 0;
    int max_count = -1;
    int min_count = INT_MAX;
    for (const auto& entry : attribute_counter) {
        if (entry.second > max_count) {
            max_count = entry.second;
            max_attribute = entry.first;
        }

        if (entry.second < min_count) {
            min_count = entry.second;
            min_attribute = entry.first;
        }
    }
    g->max_count_attribute = max_attribute;
    g->min_count_attribute = min_attribute;
    int diff = abs(max_count - min_count);
    return diff;
}

std::vector<int> get_candidates(graph* g, int max_attribute) {
    std::vector<int> candidate = {};
    for (int i = 0; i < g->n; i++) {
        if (g->exist[i] == 1 && g->k_core[i] == 1) {
            if (g->group[i] != -1) {
                // 如果这个group还没有节点在candidate中，而且group当前节点的属性为max_attribute (如果这个group没有max_attribute的点，都不加入candidate，因为删除这个group对提升fairness没有影响)
                if (g->group_sel[g->group[i]] == 0 && g->attribute[i] == max_attribute) {
                // if (g->group_sel[g->group[i]] == 0) {
                    candidate.push_back(i);
                    g->group_sel[g->group[i]] = 1; //表示这个group中已经有节点在candidate中了
                }
            } else {
                candidate.push_back(i);
            }
        }
    }
    return candidate;
}

// Main algorithm for computing the maximum fair k-core
void FKC(graph* g, int k, int threshold) {
    // step 1: Graph structure pruning (GSP)
    compute_k_core(g, k);
    print_k_core_size(g);
    compute_attribute_counter(g);

    // step 2: Group Pruning
    group_pruning(g, k, threshold);
    // printf("Number of core-groups in the graph: %lu\n\n", g->group_sel.size());
    
    // step 3: Calculate unfairness index
    int unfairness = compute_unfairness(g, g->attribute_counter);
    std::cout << "The unfairness computed is " << unfairness << "\n";

    // step 4: Select candidate nodes
    int max_attribute = g->max_count_attribute;
    std::vector<int> candidate = get_candidates(g, max_attribute);
    // printf("Candidate size: %lu, max_attribute: %d\n\n", candidate.size(), max_attribute);
    
    // loop until the clique becomes fair
    while (unfairness > threshold) {
        // printf("-----------------------------------\n");
        print_k_core_size(g);
        // printf("Candidate size: %lu, max_attribute: %d\n", candidate.size(), max_attribute);

        if (candidate.size() == 0) {
            printf("No candidate nodes found, Fair k-core construction failed!\n");
            return;
        }

        int min_cost = g->n;
        int min_cost_node = -1;
        for (auto v : candidate) {
            g->del_cost[v] = compute_delCost(g, k, v);
            if (g->del_cost[v].size() < min_cost) {
                min_cost = g->del_cost[v].size();
                min_cost_node = v;
            }
        }

        // If no node is found, break the loop
        if (min_cost_node == -1) {
            std::cout << "Error when finding best node\n";
            return;
        }

        // printf("--- min cost node: %d, cost: %d --- \n", min_cost_node, min_cost);

        //操作1 删除需要被删除的节点，更新其他节点的正度 更新candidate
		std::queue<int> del = {};
		for (int i = 0; i < g->del_cost[min_cost_node].size(); i++)//把delCost[best_n]中的所有节点都删除并且更新他们邻居的度
		{
			int id = g->del_cost[min_cost_node][i];
            del.push(id);
			delete_node(g, id, true);   //这个节点要被删除了，所以exist置0，并且更新attribute counter
			int position = BinarySearch(candidate, id);
			if (position != -1)
				candidate.erase(candidate.begin() + position); 
		}

        std::vector<int> change_set = {};
        while (!del.empty()) {
			int q = del.front();
			del.pop();
			for (int i = 0; i < g->adj_list[q].size(); i++) {
				int nid = g->adj_list[q][i];//nid是这个邻居的名称
				if (g->exist[nid] == 1)//如果这个邻居还没有被删除，就把这个邻居的度减1
				{
					if (--g->degree[nid] < k)
					{
						del.push(nid);
						delete_node(g, nid, true);  //这个节点要被删除了，所以exist置0，并且更新attribute counter
					} else {
                        change_set.push_back(nid);  // 收集delCost会发生改变的节点
                    }
				}
			}
		}

        // Remove duplicates
        std::sort(change_set.begin(), change_set.end());
		change_set.erase(std::unique(change_set.begin(), change_set.end()), change_set.end());

        for (auto id : change_set)//对changeSet中节点的相关信息进行更新
		{
			std::vector<int> new_delCost = compute_delCost(g, k, id);//该节点新的deletion cost
			g->del_cost[id] = new_delCost;
        }
        
        // check if the result is still a k-core
        bool flag = is_k_core(g, k);

        if (! flag) {
            std::cout << "After deleting, no Fair k-core preserved with unfairness threshold " << threshold << "\n";
            return;
        }

        // Update unfairness 
        unfairness = compute_unfairness(g, g->attribute_counter);
    }
    std::cout << "Fair k-core constructed successfully! Final fair ";
    print_k_core_size(g);
}

void FKC2(graph* g, int k, int threshold) {
    // step 1: Graph structure pruning (GSP)
    // compute_k_core(g, k);

    // compute_attribute_counter(g);

    // step 2: Group Pruning
    group_pruning(g, k, threshold);
    printf("Number of core-groups in the graph: %lu\n\n", g->group_sel.size());
    
    // step 3: Calculate unfairness index
    int unfairness = compute_unfairness(g, g->attribute_counter);
    std::cout << "The unfairness computed is " << unfairness << "\n";

    // step 4: Select candidate nodes
    int max_attribute = g->max_count_attribute;
    std::vector<int> candidate = get_candidates(g, max_attribute);
    printf("Candidate size: %lu, max_attribute: %d\n\n", candidate.size(), max_attribute);
    
    // loop until the clique becomes fair
    while (unfairness > threshold) {
        printf("-----------------------------------\n");
        print_k_core_size(g);
        printf("Candidate size: %lu, max_attribute: %d\n", candidate.size(), max_attribute);

        if (candidate.size() == 0) {
            printf("No candidate nodes found, Fair k-core construction failed!\n");
            return;
        }

        int min_cost = g->n;
        int min_cost_node = -1;
        for (auto v : candidate) {
            g->del_cost[v] = compute_delCost(g, k, v);
            if (g->del_cost[v].size() < min_cost) {
                min_cost = g->del_cost[v].size();
                min_cost_node = v;
            }
        }

        // If no node is found, break the loop
        if (min_cost_node == -1) {
            std::cout << "Error when finding best node\n";
            return;
        }

        printf("--- min cost node: %d, cost: %d --- \n", min_cost_node, min_cost);

        //操作1 删除需要被删除的节点，更新其他节点的正度 更新candidate
		std::queue<int> del = {};
		for (int i = 0; i < g->del_cost[min_cost_node].size(); i++)//把delCost[best_n]中的所有节点都删除并且更新他们邻居的度
		{
			int id = g->del_cost[min_cost_node][i];
            del.push(id);
			delete_node(g, id, true);   //这个节点要被删除了，所以exist置0，并且更新attribute counter
			int position = BinarySearch(candidate, id);
			if (position != -1)
				candidate.erase(candidate.begin() + position); 
		}

        std::vector<int> change_set = {};
        while (!del.empty()) {
			int q = del.front();
			del.pop();
			for (int i = 0; i < g->adj_list[q].size(); i++) {
				int nid = g->adj_list[q][i];//nid是这个邻居的名称
				if (g->exist[nid] == 1)//如果这个邻居还没有被删除，就把这个邻居的度减1
				{
					if (--g->degree[nid] < k)
					{
						del.push(nid);
						delete_node(g, nid, true);  //这个节点要被删除了，所以exist置0，并且更新attribute counter
					} else {
                        change_set.push_back(nid);  // 收集delCost会发生改变的节点
                    }
				}
			}
		}

        // Remove duplicates
        std::sort(change_set.begin(), change_set.end());
		change_set.erase(std::unique(change_set.begin(), change_set.end()), change_set.end());

        for (auto id : change_set)//对changeSet中节点的相关信息进行更新
		{
			std::vector<int> new_delCost = compute_delCost(g, k, id);//该节点新的deletion cost
			g->del_cost[id] = new_delCost;
        }
        
        // check if the result is still a k-core
        bool flag = is_k_core(g, k);

        if (! flag) {
            std::cout << "After deleting, no Fair k-core preserved with unfairness threshold " << threshold << "\n";
            return;
        }

        // Update unfairness 
        unfairness = compute_unfairness(g, g->attribute_counter);
    }
    std::cout << "Fair k-core constructed successfully! Final fair ";
    print_k_core_size(g);
}

void FKC_on_components(graph* g, int k, int threshold) {
    // Compute k-core on the whole graph
    compute_k_core(g, k);
    // Compute connected components
    std::vector<std::vector<int>> components;
    get_connected_components(g, components);
    printf("Number of connected components in the graph: %lu\n", components.size());
    // Iterate over each connected component (each k-core)
    int counter = 0;
    for (const auto& component : components) {
        if (component.size() < k+1) {
            continue;
        }
        // Get the subgraph corresponding to the connected component
        graph* sub_g = get_subgraph(g, component);
        compute_attribute_counter(sub_g);
        printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        // Call the FKC2 function on the subgraph
        FKC2(sub_g, k, threshold);
        if (is_k_core(sub_g, k) == true) {
            printf("\nFinish computing Fair k-core for component %d\n\n", counter++);
        } else {
            printf("\nNo qualified fair k-core for component %d\n\n", counter++);
        }
        // Clean up the subgraph memory
        delete sub_g;
    }
}

graph* get_subgraph(graph* g, const std::vector<int>& component) {
    graph* sub_g = new graph();
    int n = component.size();
    sub_g->n = n;
    sub_g->e = g->e;
    
    // Create a mapping from original vertex indices to subgraph indices
    std::unordered_map<int, int> vertex_mapping;
    for (int i = 0; i < n; ++i) {
        vertex_mapping[component[i]] = i;
    }

    // Initialize subgraph data structures
    sub_g->n = n;
    sub_g->adj_list.resize(n);
    sub_g->del_cost.resize(n);

    sub_g->exist.resize(n, 1);
    sub_g->degree.resize(n, 0);
    sub_g->k_core.resize(n, 1);
    sub_g->group.resize(n, -1);

    sub_g->attribute.resize(n);
    // Populate subgraph data structures
    for (int i = 0; i < n; ++i) {
        int u = component[i];
        sub_g->attribute[i] = g->attribute[u];
        // Add neighbors to subgraph adjacency list
        for (int v : g->adj_list[u]) {
            if (vertex_mapping.count(v) > 0) {
                int new_v = vertex_mapping[v];
                sub_g->adj_list[i].push_back(new_v);
                sub_g->degree[i]++;
            }
        }
    }

    return sub_g;
}




void test(graph* g, int k) {
    for (int i = 0; i < g->n; ++i) {
        if (g->exist[i] == 1) {
            g->del_cost[i] = compute_delCost(g, k, i);
        }
    }
}

// Delete one single node and vertice counter for its attribute
void delete_node(graph* g, int node, bool update_attribute) {
    if (g->exist[node] == 0 || g->k_core[node] == 0) {
        std::cout << "ERR: The node " << node << " does not exist !";
        return;
    }

    g->exist[node] = 0;
    g->k_core[node] = 0;
    // update the vertices count for attribute
    if (update_attribute) {
        int attr = g->attribute[node];
        if (g->attribute_counter.find(attr) != g->attribute_counter.end()) {
            g->attribute_counter[attr]--;
        } else {
            std::cout << "Error when delete node\n";
        }
    }
}

// Delete the nodes in cascade based on delcost of the node
void delete_nodes(graph* g, int node, bool update_attribute) {
    if (g->del_cost[node].size() < 1) {
        std::cout << "ERR: The del-cost of the node " << node << " has not been computed";
        return;
    }

    if (update_attribute) {
        for (int del_node : g->del_cost[node]) {
            delete_node(g, del_node, true);
        }
    }   
}


// Delete the nodes in cascade due to the removal of node
void delete_nodes_cascade(graph* g, int k, int node, bool update_attribute) {
    delete_node(g, node, true);
    std::queue<int> del_q;
    del_q.push(node);

    while (!del_q.empty()) {
        int u = del_q.front();
        del_q.pop();
        for (int v : g->adj_list[u]) {
            if (g->exist[v] && --g->degree[v] < k) {
                delete_node(g, v, true);
                del_q.push(v);
            }
        }
    }
}

// Recursive way to avoid stack overflow when the graph is large
void DFS(int u, graph* g, std::vector<bool>& visited, std::vector<int>& comp) {
    std::stack<int> stk;
    stk.push(u);

    while (!stk.empty()) {
        int current = stk.top();
        stk.pop();

        if (visited[current]) {
            continue;
        }

        visited[current] = true;
        comp.push_back(current);

        for (int v : g->adj_list[current]) {
            if (g->exist[v] == 1 && !visited[v]) {
                stk.push(v);
            }
        }
    }
}

void get_connected_components(graph* g, std::vector<std::vector<int>>& components) {
    std::vector<bool> visited(g->n, false);  // initialize visited array to false

    // Check if adjacency list is empty
    if (g->n == 0) {
        printf("ERR: Empty Graph\n");
        return;
    }

    for (int u = 0; u < g->n; ++u) {
        if (g->exist[u] == 1 && !visited[u]) {  // if u has not been visited yet
            std::vector<int> comp;  // create a new component
            DFS(u, g, visited, comp);  // find all vertices in the component
            components.push_back(comp);  // add the component to the list
        }
    }
}


void print_graph(graph* g) {
    for(int i = 0; i < g->n; i++) {
        std::cout << "Node " << i << ": ";
        for(int j = 0; j < g->degree[i]; j++) {
            std::cout << g->adj_list[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<int> getUnion(std::vector<int> A, std::vector<int> B)
{
	std::vector<int> result = {};
	int i = 0, j = 0;
	while (i < A.size() && j < B.size())
	{
		if (A[i] > B[j])
		{
			result.push_back(B[j]);
			j++;
		}
		else if (A[i] < B[j])
		{
			result.push_back(A[i]);
			i++;
		}
		else
		{
			result.push_back(A[i]);
			i++;
			j++;
		}
	}
	while (i < A.size())
	{
		result.push_back(A[i]);
		i++;
	}
	while (j < B.size())
	{
		result.push_back(B[j]);
		j++;
	}
	return result;
}

std::vector<int> getDifference(std::vector<int> A, std::vector<int> B)
{
	std::vector<int> result = {};
	int i = 0, j = 0;
	while (i < A.size() && j < B.size())
	{
		if (A[i] > B[j])
			j++;
		else if (A[i] < B[j])
		{
			result.push_back(A[i]);
			i++;
		}
		else
		{
			i++;
			j++;
		}
	}
	while (i < A.size())
	{
		result.push_back(A[i]);
		i++;
	}
	return result;
}

int BinarySearch(std::vector<int> &a, int value)
{
	int low, high, mid, n = a.size();
	low = 0;
	high = n - 1;
	while (low <= high)
	{
		mid = (low + high) / 2;
		if (a[mid] == value)
			return mid;
		if (a[mid] > value)
			high = mid - 1;
		if (a[mid] < value)
			low = mid + 1;
	} 
	return -1;
}

int BinarySearch1(std::vector<int> &a, int value)
{
	int low, high, mid, n = a.size();
	low = 0;
	high = n - 1;
	while (low <= high)
	{
		mid = (low + high) / 2;
		if (a[mid] == value)
			return 1;
		if (a[mid] > value)
			high = mid - 1;
		if (a[mid] < value)
			low = mid + 1;
	}
	return 0;
}

void InsertSort(std::vector<int> &a)
{
	int n = a.size();
	for (int j = 1; j < n; j++)
	{
		int key = a[j]; //待排序第一个元素
		int i = j - 1;  //代表已经排过序的元素最后一个索引数
		while (i >= 0 && key < a[i])
		{
			//从后向前逐个比较已经排序过数组，如果比它小，则把后者用前者代替，
			//其实说白了就是数组逐个后移动一位,为找到合适的位置时候便于Key的插入
			a[i + 1] = a[i];
			i--;
		}
		a[i + 1] = key;//找到合适的位置了，赋值,在i索引的后面设置key值。
	}
}

void print_k_core(graph* g) {
    std::cout << "k-core: [";
    for (int i = 0; i < g->n; ++i) {
        if (g->k_core[i] == 1) {
            std::cout << " " << i;
        }
    }
    std::cout << "]\n";
}

int get_k_core_size(graph* g) {
    int counter = 0;
    std::cout << "k-core size: ";
    for (int i = 0; i < g->n; ++i) {
        if (g->exist[i] == 1 && g->k_core[i] == 1) {
            counter++;
        }
    }
    return counter;
}

  
void print_k_core_size(graph* g) {
    int counter = 0;
    std::cout << "k-core size: ";
    for (int i = 0; i < g->n; ++i) {
        if (g->exist[i] == 1 && g->k_core[i] == 1) {
            counter++;
        }
    }
    std::cout << counter << "\n";
}

int main(int argc, char** argv) {
	if (argc < 5) {
		printf("Not enough args: ./graph k threshold edgelist.txt attributes.txt \n");
		return 0;
	}

    int k = atoi(argv[1]);
    int threshold = atoi(argv[2]);

	printf("Reading graph from file %s in edgelist format\n", argv[3]);
	graph* g = read_edgelist(argv[3]);
    
    printf("Reading attribute file from file %s\n", argv[4]);
    read_attribute(g, argv[4]);
    
    printf("Building the maximum fair %d-core structure... \n\n", k);
    FKC(g, k, threshold);
    // FKC_on_components(g, k, threshold);

    // printf("Start checking degree contraint....\n");
    // for (int i = 0; i < g->n; i++) {
    //     if (g->exist[i] == 1 && g->degree[i] < k) {
    //         printf("Not K-core !\n");
    //         break;
    //     }
    // }
    // printf("Start checking Fairness contraint: %d\n", compute_unfairness(g, g->attribute_counter));

    // ========================= TEST ==============================
    // compute_k_core(g, k);
    // print_k_core_size(g);
    // compute_attribute_counter(g);
    // int unfairness = compute_unfairness(g, g->attribute_counter);
    // printf("unfairness %d\n", unfairness);

    // compute_k_core(g, k);
    // printf("\nBuilding the k-core done... \n");
    // print_k_core_size(g);
    // test(g, k);

    // group_pruning(g, k, threshold);
    // for (int i = 0; i < g->group_vs.size(); i++) {
    //     printf("Group %d: ", i);
    //     for (auto v : g->group_vs[i]) {
    //         printf("%d ", v);
    //     }
    //     printf("\n");
    // }

	return 0;
}