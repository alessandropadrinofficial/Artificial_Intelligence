<!-- This file is machine generated: DO NOT EDIT! -->
<!-- common_typos_disable -->

# graph_nets - module reference
Common network architectures implemented as Sonnet modules.

## Other Functions and Classes
### [`class blocks.EdgeBlock`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?q=class:EdgeBlock)<!-- blocks.EdgeBlock .code-reference -->

Edge block.

A block that updates the features of each edge in a batch of graphs based on
(a subset of) the previous edge features, the features of the adjacent nodes,
and the global features of the corresponding graph.

See https://arxiv.org/abs/1806.01261 for more details.

#### [`blocks.EdgeBlock.__init__(edge_model_fn, use_edges=True, use_receiver_nodes=True, use_sender_nodes=True, use_globals=True, name='edge_block')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=374)<!-- blocks.EdgeBlock.__init__ .code-reference -->

Initializes the EdgeBlock module.

##### Args:


* `edge_model_fn`: A callable that will be called in the variable scope of
    this EdgeBlock and should return a Sonnet module (or equivalent
    callable) to be used as the edge model. The returned module should take
    a `Tensor` (of concatenated input features for each edge) and return a
    `Tensor` (of output features for each edge). Typically, this module
    would input and output `Tensor`s of rank 2, but it may also be input or
    output larger ranks. See the `_build` method documentation for more
    details on the acceptable inputs to this module in that case.
* `use_edges`: (bool, default=True). Whether to condition on edge attributes.
* `use_receiver_nodes`: (bool, default=True). Whether to condition on receiver
    node attributes.
* `use_sender_nodes`: (bool, default=True). Whether to condition on sender
    node attributes.
* `use_globals`: (bool, default=True). Whether to condition on global
    attributes.
* `name`: The module name.

##### Raises:


* `ValueError`: When fields that are required are missing.


#### [`blocks.EdgeBlock.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=418)<!-- blocks.EdgeBlock.__call__ .code-reference -->

Connects the edge block.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
    features (if `use_edges` is `True`), individual nodes features (if
    `use_receiver_nodes` or `use_sender_nodes` is `True`) and per graph
    globals (if `use_globals` is `True`) should be concatenable on the last
    axis.

##### Returns:

  An output `graphs.GraphsTuple` with updated edges.

##### Raises:


* `ValueError`: If `graph` does not have non-`None` receivers and senders, or
    if `graph` has `None` fields incompatible with the selected `use_edges`,
    `use_receiver_nodes`, `use_sender_nodes`, or `use_globals` options.


#### [`blocks.EdgeBlock.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`blocks.EdgeBlock.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`blocks.EdgeBlock.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`blocks.EdgeBlock.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgeBlock.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`blocks.EdgeBlock.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgeBlock.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`blocks.EdgeBlock.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`blocks.EdgeBlock.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgeBlock.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.module_name .code-reference -->

Returns the name of the Module.


#### [`blocks.EdgeBlock.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`blocks.EdgeBlock.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgeBlock.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`blocks.EdgeBlock.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgeBlock.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgeBlock.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgeBlock.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class blocks.EdgesToGlobalsAggregator`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?q=class:EdgesToGlobalsAggregator)<!-- blocks.EdgesToGlobalsAggregator .code-reference -->

Aggregates all edges into globals.

#### [`blocks.EdgesToGlobalsAggregator.__init__(reducer, name='edges_to_globals_aggregator')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=164)<!-- blocks.EdgesToGlobalsAggregator.__init__ .code-reference -->

Initializes the EdgesToGlobalsAggregator module.

The reducer is used for combining per-edge features (one set of edge
feature vectors per graph) to give per-graph features (one feature
vector per graph). The reducer should take a `Tensor` of edge features, a
`Tensor` of segment indices, and a number of graphs. It should be invariant
under permutation of edge features within each graph.

Examples of compatible reducers are:
* tf.unsorted_segment_sum
* tf.unsorted_segment_mean
* tf.unsorted_segment_prod
* unsorted_segment_min_or_zero
* unsorted_segment_max_or_zero

##### Args:


* `reducer`: A function for reducing sets of per-edge features to individual
    per-graph features.
* `name`: The module name.


#### [`blocks.EdgesToGlobalsAggregator.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=188)<!-- blocks.EdgesToGlobalsAggregator.__call__ .code-reference -->




#### [`blocks.EdgesToGlobalsAggregator.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`blocks.EdgesToGlobalsAggregator.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`blocks.EdgesToGlobalsAggregator.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`blocks.EdgesToGlobalsAggregator.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgesToGlobalsAggregator.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`blocks.EdgesToGlobalsAggregator.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgesToGlobalsAggregator.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`blocks.EdgesToGlobalsAggregator.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`blocks.EdgesToGlobalsAggregator.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgesToGlobalsAggregator.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.module_name .code-reference -->

Returns the name of the Module.


#### [`blocks.EdgesToGlobalsAggregator.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`blocks.EdgesToGlobalsAggregator.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgesToGlobalsAggregator.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`blocks.EdgesToGlobalsAggregator.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgesToGlobalsAggregator.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.EdgesToGlobalsAggregator.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.EdgesToGlobalsAggregator.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class blocks.GlobalBlock`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?q=class:GlobalBlock)<!-- blocks.GlobalBlock .code-reference -->

Global block.

A block that updates the global features of each graph in a batch based on
(a subset of) the previous global features, the aggregated features of the
edges of the graph, and the aggregated features of the nodes of the graph.

See https://arxiv.org/abs/1806.01261 for more details.

#### [`blocks.GlobalBlock.__init__(global_model_fn, use_edges=True, use_nodes=True, use_globals=True, nodes_reducer=unsorted_segment_sum, edges_reducer=unsorted_segment_sum, name='global_block')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=579)<!-- blocks.GlobalBlock.__init__ .code-reference -->

Initializes the GlobalBlock module.

##### Args:


* `global_model_fn`: A callable that will be called in the variable scope of
    this GlobalBlock and should return a Sonnet module (or equivalent
    callable) to be used as the global model. The returned module should
    take a `Tensor` (of concatenated input features) and return a `Tensor`
    (the global output features). Typically, this module would input and
    output `Tensor`s of rank 2, but it may also input or output larger
    ranks. See the `_build` method documentation for more details on the
    acceptable inputs to this module in that case.
* `use_edges`: (bool, default=True) Whether to condition on aggregated edges.
* `use_nodes`: (bool, default=True) Whether to condition on node attributes.
* `use_globals`: (bool, default=True) Whether to condition on global
    attributes.
* `nodes_reducer`: Reduction to be used when aggregating nodes. This should
    be a callable whose signature matches tf.unsorted_segment_sum.
* `edges_reducer`: Reduction to be used when aggregating edges. This should
    be a callable whose signature matches tf.unsorted_segment_sum.
* `name`: The module name.

##### Raises:


* `ValueError`: When fields that are required are missing.


#### [`blocks.GlobalBlock.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=637)<!-- blocks.GlobalBlock.__call__ .code-reference -->

Connects the global block.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
    (if `use_edges` is `True`), individual nodes (if `use_nodes` is True)
    and per graph globals (if `use_globals` is `True`) should be
    concatenable on the last axis.

##### Returns:

  An output `graphs.GraphsTuple` with updated globals.


#### [`blocks.GlobalBlock.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`blocks.GlobalBlock.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`blocks.GlobalBlock.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`blocks.GlobalBlock.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.GlobalBlock.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`blocks.GlobalBlock.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.GlobalBlock.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`blocks.GlobalBlock.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`blocks.GlobalBlock.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.GlobalBlock.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.module_name .code-reference -->

Returns the name of the Module.


#### [`blocks.GlobalBlock.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`blocks.GlobalBlock.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.GlobalBlock.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`blocks.GlobalBlock.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.GlobalBlock.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.GlobalBlock.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.GlobalBlock.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class blocks.NodeBlock`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?q=class:NodeBlock)<!-- blocks.NodeBlock .code-reference -->

Node block.

A block that updates the features of each node in batch of graphs based on
(a subset of) the previous node features, the aggregated features of the
adjacent edges, and the global features of the corresponding graph.

See https://arxiv.org/abs/1806.01261 for more details.

#### [`blocks.NodeBlock.__init__(node_model_fn, use_received_edges=True, use_sent_edges=False, use_nodes=True, use_globals=True, received_edges_reducer=unsorted_segment_sum, sent_edges_reducer=unsorted_segment_sum, name='node_block')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=469)<!-- blocks.NodeBlock.__init__ .code-reference -->

Initializes the NodeBlock module.

##### Args:


* `node_model_fn`: A callable that will be called in the variable scope of
    this NodeBlock and should return a Sonnet module (or equivalent
    callable) to be used as the node model. The returned module should take
    a `Tensor` (of concatenated input features for each node) and return a
    `Tensor` (of output features for each node). Typically, this module
    would input and output `Tensor`s of rank 2, but it may also be input or
    output larger ranks. See the `_build` method documentation for more
    details on the acceptable inputs to this module in that case.
* `use_received_edges`: (bool, default=True) Whether to condition on
    aggregated edges received by each node.
* `use_sent_edges`: (bool, default=False) Whether to condition on aggregated
    edges sent by each node.
* `use_nodes`: (bool, default=True) Whether to condition on node attributes.
* `use_globals`: (bool, default=True) Whether to condition on global
    attributes.
* `received_edges_reducer`: Reduction to be used when aggregating received
    edges. This should be a callable whose signature matches
    `tf.unsorted_segment_sum`.
* `sent_edges_reducer`: Reduction to be used when aggregating sent edges.
    This should be a callable whose signature matches
    `tf.unsorted_segment_sum`.
* `name`: The module name.

##### Raises:


* `ValueError`: When fields that are required are missing.


#### [`blocks.NodeBlock.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=536)<!-- blocks.NodeBlock.__call__ .code-reference -->

Connects the node block.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
    features (if `use_received_edges` or `use_sent_edges` is `True`),
    individual nodes features (if `use_nodes` is True) and per graph globals
    (if `use_globals` is `True`) should be concatenable on the last axis.

##### Returns:

  An output `graphs.GraphsTuple` with updated nodes.


#### [`blocks.NodeBlock.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`blocks.NodeBlock.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`blocks.NodeBlock.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`blocks.NodeBlock.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodeBlock.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`blocks.NodeBlock.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodeBlock.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`blocks.NodeBlock.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`blocks.NodeBlock.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodeBlock.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.module_name .code-reference -->

Returns the name of the Module.


#### [`blocks.NodeBlock.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`blocks.NodeBlock.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodeBlock.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`blocks.NodeBlock.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodeBlock.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodeBlock.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodeBlock.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class blocks.NodesToGlobalsAggregator`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?q=class:NodesToGlobalsAggregator)<!-- blocks.NodesToGlobalsAggregator .code-reference -->

Aggregates all nodes into globals.

#### [`blocks.NodesToGlobalsAggregator.__init__(reducer, name='nodes_to_globals_aggregator')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=200)<!-- blocks.NodesToGlobalsAggregator.__init__ .code-reference -->

Initializes the NodesToGlobalsAggregator module.

The reducer is used for combining per-node features (one set of node
feature vectors per graph) to give per-graph features (one feature
vector per graph). The reducer should take a `Tensor` of node features, a
`Tensor` of segment indices, and a number of graphs. It should be invariant
under permutation of node features within each graph.

Examples of compatible reducers are:
* tf.unsorted_segment_sum
* tf.unsorted_segment_mean
* tf.unsorted_segment_prod
* unsorted_segment_min_or_zero
* unsorted_segment_max_or_zero

##### Args:


* `reducer`: A function for reducing sets of per-node features to individual
    per-graph features.
* `name`: The module name.


#### [`blocks.NodesToGlobalsAggregator.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=224)<!-- blocks.NodesToGlobalsAggregator.__call__ .code-reference -->




#### [`blocks.NodesToGlobalsAggregator.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`blocks.NodesToGlobalsAggregator.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`blocks.NodesToGlobalsAggregator.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`blocks.NodesToGlobalsAggregator.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodesToGlobalsAggregator.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`blocks.NodesToGlobalsAggregator.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodesToGlobalsAggregator.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`blocks.NodesToGlobalsAggregator.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`blocks.NodesToGlobalsAggregator.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodesToGlobalsAggregator.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.module_name .code-reference -->

Returns the name of the Module.


#### [`blocks.NodesToGlobalsAggregator.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`blocks.NodesToGlobalsAggregator.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodesToGlobalsAggregator.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`blocks.NodesToGlobalsAggregator.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodesToGlobalsAggregator.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.NodesToGlobalsAggregator.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.NodesToGlobalsAggregator.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class blocks.ReceivedEdgesToNodesAggregator`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?q=class:ReceivedEdgesToNodesAggregator)<!-- blocks.ReceivedEdgesToNodesAggregator .code-reference -->

Agregates received edges into the corresponding receiver nodes.

#### [`blocks.ReceivedEdgesToNodesAggregator.__init__(reducer, name='received_edges_to_nodes_aggregator')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=283)<!-- blocks.ReceivedEdgesToNodesAggregator.__init__ .code-reference -->

Constructor.

The reducer is used for combining per-edge features (one set of edge
feature vectors per node) to give per-node features (one feature
vector per node). The reducer should take a `Tensor` of edge features, a
`Tensor` of segment indices, and a number of nodes. It should be invariant
under permutation of edge features within each segment.

Examples of compatible reducers are:
* tf.unsorted_segment_sum
* tf.unsorted_segment_mean
* tf.unsorted_segment_prod
* unsorted_segment_min_or_zero
* unsorted_segment_max_or_zero

##### Args:


* `reducer`: A function for reducing sets of per-edge features to individual
    per-node features.
* `name`: The module name.


#### [`blocks.ReceivedEdgesToNodesAggregator.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=242)<!-- blocks.ReceivedEdgesToNodesAggregator.__call__ .code-reference -->




#### [`blocks.ReceivedEdgesToNodesAggregator.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`blocks.ReceivedEdgesToNodesAggregator.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`blocks.ReceivedEdgesToNodesAggregator.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`blocks.ReceivedEdgesToNodesAggregator.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.ReceivedEdgesToNodesAggregator.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`blocks.ReceivedEdgesToNodesAggregator.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.ReceivedEdgesToNodesAggregator.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`blocks.ReceivedEdgesToNodesAggregator.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`blocks.ReceivedEdgesToNodesAggregator.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.ReceivedEdgesToNodesAggregator.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.module_name .code-reference -->

Returns the name of the Module.


#### [`blocks.ReceivedEdgesToNodesAggregator.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`blocks.ReceivedEdgesToNodesAggregator.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.ReceivedEdgesToNodesAggregator.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`blocks.ReceivedEdgesToNodesAggregator.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.ReceivedEdgesToNodesAggregator.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.ReceivedEdgesToNodesAggregator.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.ReceivedEdgesToNodesAggregator.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class blocks.SentEdgesToNodesAggregator`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?q=class:SentEdgesToNodesAggregator)<!-- blocks.SentEdgesToNodesAggregator .code-reference -->

Agregates sent edges into the corresponding sender nodes.

#### [`blocks.SentEdgesToNodesAggregator.__init__(reducer, name='sent_edges_to_nodes_aggregator')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=253)<!-- blocks.SentEdgesToNodesAggregator.__init__ .code-reference -->

Constructor.

The reducer is used for combining per-edge features (one set of edge
feature vectors per node) to give per-node features (one feature
vector per node). The reducer should take a `Tensor` of edge features, a
`Tensor` of segment indices, and a number of nodes. It should be invariant
under permutation of edge features within each segment.

Examples of compatible reducers are:
* tf.unsorted_segment_sum
* tf.unsorted_segment_mean
* tf.unsorted_segment_prod
* unsorted_segment_min_or_zero
* unsorted_segment_max_or_zero

##### Args:


* `reducer`: A function for reducing sets of per-edge features to individual
    per-node features.
* `name`: The module name.


#### [`blocks.SentEdgesToNodesAggregator.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=242)<!-- blocks.SentEdgesToNodesAggregator.__call__ .code-reference -->




#### [`blocks.SentEdgesToNodesAggregator.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`blocks.SentEdgesToNodesAggregator.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`blocks.SentEdgesToNodesAggregator.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`blocks.SentEdgesToNodesAggregator.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.SentEdgesToNodesAggregator.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`blocks.SentEdgesToNodesAggregator.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.SentEdgesToNodesAggregator.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`blocks.SentEdgesToNodesAggregator.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`blocks.SentEdgesToNodesAggregator.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.SentEdgesToNodesAggregator.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.module_name .code-reference -->

Returns the name of the Module.


#### [`blocks.SentEdgesToNodesAggregator.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`blocks.SentEdgesToNodesAggregator.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.SentEdgesToNodesAggregator.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`blocks.SentEdgesToNodesAggregator.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.SentEdgesToNodesAggregator.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`blocks.SentEdgesToNodesAggregator.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- blocks.SentEdgesToNodesAggregator.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`blocks.broadcast_globals_to_edges(graph, name='broadcast_globals_to_edges')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=68)<!-- blocks.broadcast_globals_to_edges .code-reference -->

Broadcasts the global features to the edges of a graph.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, with globals features of
    shape `[n_graphs] + global_shape`, and `N_EDGE` field of shape
    `[n_graphs]`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A tensor of shape `[n_edges] + global_shape`, where
  `n_edges = sum(graph.n_edge)`. The i-th element of this tensor is given by
  `globals[j]`, where j is the index of the graph the i-th edge belongs to
  (i.e. is such that
  `sum_{k < j} graphs.n_edge[k] <= i < sum_{k <= j} graphs.n_edge[k]`).

##### Raises:


* `ValueError`: If either `graph.globals` or `graph.n_edge` is `None`.


### [`blocks.broadcast_globals_to_nodes(graph, name='broadcast_globals_to_nodes')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=92)<!-- blocks.broadcast_globals_to_nodes .code-reference -->

Broadcasts the global features to the nodes of a graph.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, with globals features of
    shape `[n_graphs] + global_shape`, and `N_NODE` field of shape
    `[n_graphs]`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A tensor of shape `[n_nodes] + global_shape`, where
  `n_nodes = sum(graph.n_node)`. The i-th element of this tensor is given by
  `globals[j]`, where j is the index of the graph the i-th node belongs to
  (i.e. is such that
  `sum_{k < j} graphs.n_node[k] <= i < sum_{k <= j} graphs.n_node[k]`).

##### Raises:


* `ValueError`: If either `graph.globals` or `graph.n_node` is `None`.


### [`blocks.broadcast_receiver_nodes_to_edges(graph, name='broadcast_receiver_nodes_to_edges')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=139)<!-- blocks.broadcast_receiver_nodes_to_edges .code-reference -->

Broadcasts the node features to the edges they are receiving from.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, with nodes features of
    shape `[n_nodes] + node_shape`, and receivers of shape `[n_edges]`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A tensor of shape `[n_edges] + node_shape`, where
  `n_edges = sum(graph.n_edge)`. The i-th element is given by
  `graph.nodes[graph.receivers[i]]`.

##### Raises:


* `ValueError`: If either `graph.nodes` or `graph.receivers` is `None`.


### [`blocks.broadcast_sender_nodes_to_edges(graph, name='broadcast_sender_nodes_to_edges')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=116)<!-- blocks.broadcast_sender_nodes_to_edges .code-reference -->

Broadcasts the node features to the edges they are sending into.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, with nodes features of
    shape `[n_nodes] + node_shape`, and `senders` field of shape
    `[n_edges]`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A tensor of shape `[n_edges] + node_shape`, where
  `n_edges = sum(graph.n_edge)`. The i-th element is given by
  `graph.nodes[graph.senders[i]]`.

##### Raises:


* `ValueError`: If either `graph.nodes` or `graph.senders` is `None`.


### [`blocks.unsorted_segment_max_or_zero(values, indices, num_groups, name='unsorted_segment_max_or_zero')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=342)<!-- blocks.unsorted_segment_max_or_zero .code-reference -->

Aggregates information using elementwise max.

Segments with no elements are given a "max" of zero instead of the most
negative finite value possible (which is what `tf.unsorted_segment_max` would
do).

##### Args:


* `values`: A `Tensor` of per-element features.
* `indices`: A 1-D `Tensor` whose length is equal to `values`' first dimension.
* `num_groups`: A `Tensor`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A `Tensor` of the same type as `values`.


### [`blocks.unsorted_segment_min_or_zero(values, indices, num_groups, name='unsorted_segment_min_or_zero')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py?l=320)<!-- blocks.unsorted_segment_min_or_zero .code-reference -->

Aggregates information using elementwise min.

Segments with no elements are given a "min" of zero instead of the most
positive finite value possible (which is what `tf.unsorted_segment_min`
would do).

##### Args:


* `values`: A `Tensor` of per-element features.
* `indices`: A 1-D `Tensor` whose length is equal to `values`' first dimension.
* `num_groups`: A `Tensor`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A `Tensor` of the same type as `values`.


### [`class graphs.GraphsTuple`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/graphs.py?q=class:GraphsTuple)<!-- graphs.GraphsTuple .code-reference -->

Default namedtuple describing `Graphs`s.

A children of `collections.namedtuple`s, which allows it to be directly input
and output from `tensorflow.Session.run()` calls

#### [`graphs.GraphsTuple.__init__(*args, **kwargs)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/graphs.py?l=153)<!-- graphs.GraphsTuple.__init__ .code-reference -->




#### `graphs.GraphsTuple.edges`<!-- graphs.GraphsTuple.edges .code-reference -->

Alias for field number 1


#### `graphs.GraphsTuple.globals`<!-- graphs.GraphsTuple.globals .code-reference -->

Alias for field number 4


#### [`graphs.GraphsTuple.map(field_fn, fields=('nodes', 'edges', 'globals'))`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/graphs.py?l=165)<!-- graphs.GraphsTuple.map .code-reference -->

Applies `field_fn` to the fields `fields` of the instance.

`field_fn` is applied exactly once per field in `fields`. The result must
satisfy the `GraphsTuple` requirement w.r.t. `None` fields, i.e. the
`SENDERS` cannot be `None` if the `EDGES` or `RECEIVERS` are not `None`,
etc.

##### Args:


* `field_fn`: A callable that take a single argument.
* `fields`: (iterable of `str`). An iterable of the fields to apply
    `field_fn` to.

##### Returns:

  A copy of the instance, with the fields in `fields` replaced by the result
  of applying `field_fn` to them.


#### `graphs.GraphsTuple.n_edge`<!-- graphs.GraphsTuple.n_edge .code-reference -->

Alias for field number 6


#### `graphs.GraphsTuple.n_node`<!-- graphs.GraphsTuple.n_node .code-reference -->

Alias for field number 5


#### `graphs.GraphsTuple.nodes`<!-- graphs.GraphsTuple.nodes .code-reference -->

Alias for field number 0


#### `graphs.GraphsTuple.receivers`<!-- graphs.GraphsTuple.receivers .code-reference -->

Alias for field number 2


#### [`graphs.GraphsTuple.replace(**kwargs)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/graphs.py?l=160)<!-- graphs.GraphsTuple.replace .code-reference -->




#### `graphs.GraphsTuple.senders`<!-- graphs.GraphsTuple.senders .code-reference -->

Alias for field number 3



### [`class modules.CommNet`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?q=class:CommNet)<!-- modules.CommNet .code-reference -->

CommNet module.

Implementation for the model originally described in
https://arxiv.org/abs/1605.07736 (S. Sukhbaatar, A. Szlam, R. Fergus), in the
version presented in https://arxiv.org/abs/1706.06122 (Y. Hoshen).

This module internally creates edge features based on the features from the
nodes sending to that edge, and independently learns an embedding for each
node. It then uses these edges and nodes features to compute updated node
features.

This module does not use the global nor the edges features of the input, but
uses its receivers and senders information. The output graph has the same
value in edge and global fields as the input graph. The edge and global
features fields may have a `None` value in the input `gn_graphs.GraphsTuple`.

#### [`modules.CommNet.__init__(edge_model_fn, node_encoder_model_fn, node_model_fn, reducer=unsorted_segment_sum, name='comm_net')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=470)<!-- modules.CommNet.__init__ .code-reference -->

Initializes the CommNet module.

##### Args:


* `edge_model_fn`: A callable to be passed to EdgeBlock. The callable must
    return a Sonnet module (or equivalent; see EdgeBlock for details).
* `node_encoder_model_fn`: A callable to be passed to the NodeBlock
    responsible for the first encoding of the nodes. The callable must
    return a Sonnet module (or equivalent; see NodeBlock for details). The
    shape of this module's output should match the shape of the module built
    by `edge_model_fn`, but for the first and last dimension.
* `node_model_fn`: A callable to be passed to NodeBlock. The callable must
    return a Sonnet module (or equivalent; see NodeBlock for details).
* `reducer`: Reduction to be used when aggregating the edges in the nodes.
    This should be a callable whose signature matches
    tf.unsorted_segment_sum.
* `name`: The module name.


#### [`modules.CommNet.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=521)<!-- modules.CommNet.__call__ .code-reference -->

Connects the CommNet network.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, with non-`None` nodes,
    receivers and senders.

##### Returns:

  An output `graphs.GraphsTuple` with updated nodes.

##### Raises:


* `ValueError`: if any of `graph.nodes`, `graph.receivers` or `graph.senders`
  is `None`.


#### [`modules.CommNet.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`modules.CommNet.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`modules.CommNet.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`modules.CommNet.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.CommNet.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`modules.CommNet.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.CommNet.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`modules.CommNet.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`modules.CommNet.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.CommNet.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.module_name .code-reference -->

Returns the name of the Module.


#### [`modules.CommNet.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`modules.CommNet.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.CommNet.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`modules.CommNet.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.CommNet.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.CommNet.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.CommNet.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class modules.DeepSets`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?q=class:DeepSets)<!-- modules.DeepSets .code-reference -->

DeepSets module.

Implementation for the model described in https://arxiv.org/abs/1703.06114
(M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, A. Smola).
See also PointNet (https://arxiv.org/abs/1612.00593, C. Qi, H. Su, K. Mo,
L. J. Guibas) for a related model.

This module operates on sets, which can be thought of as graphs without
edges. The nodes features are first updated based on their value and the
globals features, and new globals features are then computed based on the
updated nodes features.

Note that in the original model, only the globals are updated in the returned
graph, while this implementation also returns updated nodes.
The original model can be reproduced by writing:
```
deep_sets = DeepSets()
output = deep_sets(input)
output = input.replace(globals=output.globals)
```

This module does not use the edges data or the information contained in the
receivers or senders; the output graph has the same value in those fields as
the input graph. Those fields can also have `None` values in the input
`graphs.GraphsTuple`.

#### [`modules.DeepSets.__init__(node_model_fn, global_model_fn, reducer=unsorted_segment_sum, name='deep_sets')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=401)<!-- modules.DeepSets.__init__ .code-reference -->

Initializes the DeepSets module.

##### Args:


* `node_model_fn`: A callable to be passed to NodeBlock. The callable must
    return a Sonnet module (or equivalent; see NodeBlock for details). The
    shape of this module's output must equal the shape of the input graph's
    global features, but for the first and last axis.
* `global_model_fn`: A callable to be passed to GlobalBlock. The callable must
    return a Sonnet module (or equivalent; see GlobalBlock for details).
* `reducer`: Reduction to be used when aggregating the nodes in the globals.
    This should be a callable whose signature matches
    tf.unsorted_segment_sum.
* `name`: The module name.


#### [`modules.DeepSets.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=436)<!-- modules.DeepSets.__call__ .code-reference -->

Connects the DeepSets network.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, whose edges, senders
    or receivers properties may be `None`. The features of every node and
    global of `graph` should be concatenable on the last axis (i.e. the
    shapes of `graph.nodes` and `graph.globals` must match but for their
    first and last axis).

##### Returns:

  An output `graphs.GraphsTuple` with updated globals.


#### [`modules.DeepSets.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`modules.DeepSets.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`modules.DeepSets.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`modules.DeepSets.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.DeepSets.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`modules.DeepSets.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.DeepSets.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`modules.DeepSets.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`modules.DeepSets.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.DeepSets.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.module_name .code-reference -->

Returns the name of the Module.


#### [`modules.DeepSets.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`modules.DeepSets.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.DeepSets.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`modules.DeepSets.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.DeepSets.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.DeepSets.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.DeepSets.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class modules.GraphIndependent`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?q=class:GraphIndependent)<!-- modules.GraphIndependent .code-reference -->

A graph block that applies models to the graph elements independently.

The inputs and outputs are graphs. The corresponding models are applied to
each element of the graph (edges, nodes and globals) in parallel and
independently of the other elements. It can be used to encode or
decode the elements of a graph.

#### [`modules.GraphIndependent.__init__(edge_model_fn=None, node_model_fn=None, global_model_fn=None, name='graph_independent')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=314)<!-- modules.GraphIndependent.__init__ .code-reference -->

Initializes the GraphIndependent module.

##### Args:


* `edge_model_fn`: A callable that returns an edge model function. The
    callable must return a Sonnet module (or equivalent). If passed `None`,
    will pass through inputs (the default).
* `node_model_fn`: A callable that returns a node model function. The callable
    must return a Sonnet module (or equivalent). If passed `None`, will pass
    through inputs (the default).
* `global_model_fn`: A callable that returns a global model function. The
    callable must return a Sonnet module (or equivalent). If passed `None`,
    will pass through inputs (the default).
* `name`: The module name.


#### [`modules.GraphIndependent.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=356)<!-- modules.GraphIndependent.__call__ .code-reference -->

Connects the GraphIndependent.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing non-`None` edges, nodes and
    globals.

##### Returns:

  An output `graphs.GraphsTuple` with updated edges, nodes and globals.


#### [`modules.GraphIndependent.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`modules.GraphIndependent.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`modules.GraphIndependent.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`modules.GraphIndependent.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphIndependent.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`modules.GraphIndependent.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphIndependent.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`modules.GraphIndependent.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`modules.GraphIndependent.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphIndependent.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.module_name .code-reference -->

Returns the name of the Module.


#### [`modules.GraphIndependent.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`modules.GraphIndependent.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphIndependent.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`modules.GraphIndependent.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphIndependent.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphIndependent.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphIndependent.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class modules.GraphNetwork`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?q=class:GraphNetwork)<!-- modules.GraphNetwork .code-reference -->

Implementation of a Graph Network.

See https://arxiv.org/abs/1806.01261 for more details.

#### [`modules.GraphNetwork.__init__(edge_model_fn, node_model_fn, global_model_fn, reducer=unsorted_segment_sum, edge_block_opt=None, node_block_opt=None, global_block_opt=None, name='graph_network')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=240)<!-- modules.GraphNetwork.__init__ .code-reference -->

Initializes the GraphNetwork module.

##### Args:


* `edge_model_fn`: A callable that will be passed to EdgeBlock to perform
    per-edge computations. The callable must return a Sonnet module (or
    equivalent; see EdgeBlock for details).
* `node_model_fn`: A callable that will be passed to NodeBlock to perform
    per-node computations. The callable must return a Sonnet module (or
    equivalent; see NodeBlock for details).
* `global_model_fn`: A callable that will be passed to GlobalBlock to perform
    per-global computations. The callable must return a Sonnet module (or
    equivalent; see GlobalBlock for details).
* `reducer`: Reducer to be used by NodeBlock and GlobalBlock to
    to aggregate nodes and edges. Defaults to tf.unsorted_segment_sum.
    This will be overridden by the reducers specified in `node_block_opt`
    and `global_block_opt`, if any.
* `edge_block_opt`: Additional options to be passed to the EdgeBlock. Can
    contain keys `use_edges`, `use_receivers_nodes`, `use_senders_nodes`,
    `use_globals`. By default, those are all True but for
    `use_senders_nodes`.
* `node_block_opt`: Additional options to be passed to the NodeBlock. Can
    contain the keys `use_received_edges`, `use_nodes`, `use_globals` (all
    set to True by default), and `received_edges_reducer`,
    `sent_edges_reducer` (default to `reducer`).
* `global_block_opt`: Additional options to be passed to the GlobalBlock.
* `name`: The module name.


#### [`modules.GraphNetwork.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=289)<!-- modules.GraphNetwork.__call__ .code-reference -->

Connects the GraphNetwork.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s. Depending on the
    block options, `graph` may contain `None` fields; but with the default
    configuration, no `None` field is allowed. Moreover, when using the
    default configuration, the features of each nodes, edges and globals of
    `graph` should be concatenable on the last dimension.

##### Returns:

  An output `graphs.GraphsTuple` with updated edges, nodes and globals.


#### [`modules.GraphNetwork.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`modules.GraphNetwork.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`modules.GraphNetwork.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`modules.GraphNetwork.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphNetwork.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`modules.GraphNetwork.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphNetwork.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`modules.GraphNetwork.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`modules.GraphNetwork.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphNetwork.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.module_name .code-reference -->

Returns the name of the Module.


#### [`modules.GraphNetwork.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`modules.GraphNetwork.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphNetwork.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`modules.GraphNetwork.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphNetwork.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.GraphNetwork.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.GraphNetwork.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class modules.InteractionNetwork`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?q=class:InteractionNetwork)<!-- modules.InteractionNetwork .code-reference -->

Implementation of an Interaction Network.

An interaction networks computes interactions on the edges based on the
previous edges features, and on the features of the nodes sending into those
edges. It then updates the nodes based on the incomming updated edges.
See https://arxiv.org/abs/1612.00222 for more details.

This model does not update the graph globals, and they are allowed to be
`None`.

#### [`modules.InteractionNetwork.__init__(edge_model_fn, node_model_fn, reducer=unsorted_segment_sum, name='interaction_network')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=93)<!-- modules.InteractionNetwork.__init__ .code-reference -->

Initializes the InteractionNetwork module.

##### Args:


* `edge_model_fn`: A callable that will be passed to `EdgeBlock` to perform
    per-edge computations. The callable must return a Sonnet module (or
    equivalent; see `blocks.EdgeBlock` for details), and the shape of the
    output of this module must match the one of the input nodes, but for the
    first and last axis.
* `node_model_fn`: A callable that will be passed to `NodeBlock` to perform
    per-node computations. The callable must return a Sonnet module (or
    equivalent; see `blocks.NodeBlock` for details).
* `reducer`: Reducer to be used by NodeBlock to aggregate edges. Defaults
    to tf.unsorted_segment_sum.
* `name`: The module name.


#### [`modules.InteractionNetwork.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=124)<!-- modules.InteractionNetwork.__call__ .code-reference -->

Connects the InterationNetwork.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s. `graph.globals` can be
    `None`. The features of each node and edge of `graph` must be
    concatenable on the last axis (i.e., the shapes of `graph.nodes` and
    `graph.edges` must match but for their first and last axis).

##### Returns:

  An output `graphs.GraphsTuple` with updated edges and nodes.

##### Raises:


* `ValueError`: If any of `graph.nodes`, `graph.edges`, `graph.receivers` or
    `graph.senders` is `None`.


#### [`modules.InteractionNetwork.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`modules.InteractionNetwork.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`modules.InteractionNetwork.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`modules.InteractionNetwork.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.InteractionNetwork.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`modules.InteractionNetwork.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.InteractionNetwork.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`modules.InteractionNetwork.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`modules.InteractionNetwork.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.InteractionNetwork.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.module_name .code-reference -->

Returns the name of the Module.


#### [`modules.InteractionNetwork.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`modules.InteractionNetwork.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.InteractionNetwork.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`modules.InteractionNetwork.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.InteractionNetwork.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.InteractionNetwork.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.InteractionNetwork.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class modules.RelationNetwork`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?q=class:RelationNetwork)<!-- modules.RelationNetwork .code-reference -->

Implementation of a Relation Network.

See https://arxiv.org/abs/1706.01427 for more details.

The global and edges features of the input graph are not used, and are
allowed to be `None` (the receivers and senders properties must be present).
The output graph has updated, non-`None`, globals.

#### [`modules.RelationNetwork.__init__(edge_model_fn, global_model_fn, reducer=unsorted_segment_sum, name='relation_network')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=153)<!-- modules.RelationNetwork.__init__ .code-reference -->

Initializes the RelationNetwork module.

##### Args:


* `edge_model_fn`: A callable that will be passed to EdgeBlock to perform
    per-edge computations. The callable must return a Sonnet module (or
    equivalent; see EdgeBlock for details).
* `global_model_fn`: A callable that will be passed to GlobalBlock to perform
    per-global computations. The callable must return a Sonnet module (or
    equivalent; see GlobalBlock for details).
* `reducer`: Reducer to be used by GlobalBlock to aggregate edges. Defaults
    to tf.unsorted_segment_sum.
* `name`: The module name.


#### [`modules.RelationNetwork.__call__(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=188)<!-- modules.RelationNetwork.__call__ .code-reference -->

Connects the RelationNetwork.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s, except for the edges
    and global properties which may be `None`.

##### Returns:

  A `graphs.GraphsTuple` with updated globals.

##### Raises:


* `ValueError`: If any of `graph.nodes`, `graph.receivers` or `graph.senders`
    is `None`.


#### [`modules.RelationNetwork.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`modules.RelationNetwork.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`modules.RelationNetwork.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`modules.RelationNetwork.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.RelationNetwork.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`modules.RelationNetwork.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.RelationNetwork.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`modules.RelationNetwork.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`modules.RelationNetwork.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.RelationNetwork.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.module_name .code-reference -->

Returns the name of the Module.


#### [`modules.RelationNetwork.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`modules.RelationNetwork.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.RelationNetwork.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`modules.RelationNetwork.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.RelationNetwork.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.RelationNetwork.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.RelationNetwork.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`class modules.SelfAttention`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?q=class:SelfAttention)<!-- modules.SelfAttention .code-reference -->

Multi-head self-attention module.

The module is based on the following three papers:
 * A simple neural network module for relational reasoning (RNs):
     https://arxiv.org/abs/1706.01427
 * Non-local Neural Networks: https://arxiv.org/abs/1711.07971.
 * Attention Is All You Need (AIAYN): https://arxiv.org/abs/1706.03762.

The input to the modules consists of a graph containing values for each node
and connectivity between them, a tensor containing keys for each node
and a tensor containing queries for each node.

The self-attention step consist of updating the node values, with each new
node value computed in a two step process:
- Computing the attention weights between each node and all of its senders
 nodes, by calculating sum(sender_key*receiver_query) and using the softmax
 operation on all attention weights for each node.
- For each receiver node, compute the new node value as the weighted average
 of the values of the sender nodes, according to the attention weights.
- Nodes with no received edges, get an updated value of 0.

Values, keys and queries contain a "head" axis to compute independent
self-attention for each of the heads.

#### [`modules.SelfAttention.__init__(name='self_attention')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=620)<!-- modules.SelfAttention.__init__ .code-reference -->

Inits the module.

##### Args:


* `name`: The module name.


#### [`modules.SelfAttention.__call__(node_values, node_keys, node_queries, attention_graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py?l=629)<!-- modules.SelfAttention.__call__ .code-reference -->

Connects the multi-head self-attention module.

The self-attention is only computed according to the connectivity of the
input graphs, with receiver nodes attending to sender nodes.

##### Args:


* `node_values`: Tensor containing the values associated to each of the nodes.
    The expected shape is [total_num_nodes, num_heads, key_size].
* `node_keys`: Tensor containing the key associated to each of the nodes. The
    expected shape is [total_num_nodes, num_heads, key_size].
* `node_queries`: Tensor containing the query associated to each of the nodes.
    The expected shape is [total_num_nodes, num_heads, query_size]. The
    query size must be equal to the key size.
* `attention_graph`: Graph containing connectivity information between nodes
    via the senders and receivers fields. Node A will only attempt to attend
    to Node B if `attention_graph` contains an edge sent by Node A and
    received by Node B.

##### Returns:

  An output `graphs.GraphsTuple` with updated nodes containing the
  aggregated attended value for each of the nodes with shape
  [total_num_nodes, num_heads, value_size].

##### Raises:


* `ValueError`: if the input graph does not have edges.


#### [`modules.SelfAttention.connected_subgraphs`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.connected_subgraphs .code-reference -->

Returns the subgraphs created by this module so far.


#### [`modules.SelfAttention.defun()`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.defun .code-reference -->

Wraps this modules call method in a callable graph function.


#### [`modules.SelfAttention.defun_wrapped`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.defun_wrapped .code-reference -->

Returns boolean indicating whether this module is defun wrapped.


#### [`modules.SelfAttention.get_all_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.get_all_variables .code-reference -->

Returns all `tf.Variable`s used when the module is connected.

See the documentation for `AbstractModule._capture_variables()` for more
information.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.SelfAttention.get_possible_initializer_keys(cls)`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.get_possible_initializer_keys .code-reference -->

Returns the keys the dictionary of variable initializers may contain.

This provides the user with a way of knowing the initializer keys that are
available without having to instantiate a sonnet module. Subclasses may
override this class method if they need additional arguments to determine
what initializer keys may be provided.

##### Returns:

  Set with strings corresponding to the strings that may be passed to the
      constructor.


#### [`modules.SelfAttention.get_variables(collection='trainable_variables')`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.get_variables .code-reference -->

Returns tuple of `tf.Variable`s declared inside this module.

Note that this operates by searching this module's variable scope,
and so does not know about any modules that were constructed elsewhere but
used inside this module.

This method explicitly re-enters the Graph which this module has been
connected to.

##### Args:


* `collection`: Collection to restrict query to. By default this is
    `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
    variables such as moving averages.

##### Returns:

  A tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.SelfAttention.graph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.graph .code-reference -->

Returns the Graph instance which the module is connected to, or None.


#### [`modules.SelfAttention.is_connected`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.is_connected .code-reference -->

Returns true iff the Module been connected to the Graph at least once.


#### [`modules.SelfAttention.last_connected_subgraph`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.last_connected_subgraph .code-reference -->

Returns the last subgraph created by this module.

##### Returns:

  The last connected subgraph.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.SelfAttention.module_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.module_name .code-reference -->

Returns the name of the Module.


#### [`modules.SelfAttention.name_scopes`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.name_scopes .code-reference -->

Returns a tuple of all name_scopes generated by this module.


#### [`modules.SelfAttention.non_trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.non_trainable_variables .code-reference -->

All **non-trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.SelfAttention.scope_name`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.scope_name .code-reference -->

Returns the full name of the Module's variable scope.


#### [`modules.SelfAttention.trainable_variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.trainable_variables .code-reference -->

All **trainable** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.SelfAttention.variable_scope`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.variable_scope .code-reference -->

Returns the variable_scope declared by the module.

It is valid for library users to access the internal templated
variable_scope, but only makes sense to do so after connection. Therefore we
raise an error here if the variable_scope is requested before connection.

The only case where it does make sense to access the variable_scope before
connection is to get the post-uniquification name, which we support using
the separate .scope_name property.

##### Returns:


* `variable_scope`: `tf.VariableScope` instance of the internal `tf.Template`.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.


#### [`modules.SelfAttention.variables`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)<!-- modules.SelfAttention.variables .code-reference -->

**All** `tf.Variable`s used when the module is connected.

This property does not rely on global collections and should generally be
preferred vs. `get_variables` and `get_all_variables`.

See the documentation for `AbstractModule._capture_variables()` for more
information about what variables are captured.

##### Returns:

  A sorted (by variable name) tuple of `tf.Variable` objects.

##### Raises:


* `NotConnectedError`: If the module is not connected to the Graph.



### [`utils_np.data_dict_to_networkx(data_dict)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_np.py?l=227)<!-- utils_np.data_dict_to_networkx .code-reference -->

Returns a networkx graph that contains the stored data.

Depending on the content of `data_dict`, the returned `networkx` instance has
the following properties:

- The nodes feature are placed in the nodes attribute dictionary under the
  "features" key. If the `NODES` fields is `None`, a `None` value is placed
  here;

- If the `RECEIVERS` field is `None`, no edges are added to the graph.
  Otherwise, edges are added with the order in which they appeared in
  `data_dict` stored in the "index" field of their attributes dictionary;

- The edges features are placed in the edges attribute dictionary under the
  "features" key. If the `EDGES` field is `None`, a `None` value is placed;

- The global feature are placed under the key "features" of the graph
  property of the returned instance. If the `GLOBALS` field is `None`, a
  `None` global property is created.

##### Args:


* `data_dict`: A graph `dict` of Numpy data.

##### Returns:

  The `networkx.MultiDiGraph`.

##### Raises:


* `ValueError`: If the `NODES` field of `data_dict` contains `None`, and
    `data_dict` does not have a `N_NODE` field.


### [`utils_np.data_dicts_to_graphs_tuple(data_dicts)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_np.py?l=357)<!-- utils_np.data_dicts_to_graphs_tuple .code-reference -->

Constructs a `graphs.GraphsTuple` from an iterable of data dicts.

The graphs represented by the `data_dicts` argument are batched to form a
single instance of `graphs.GraphsTuple` containing numpy arrays.

##### Args:


* `data_dicts`: An iterable of dictionaries with keys `GRAPH_DATA_FIELDS`,
    plus, potentially, a subset of `GRAPH_NUMBER_FIELDS`. The NODES and
    EDGES fields should be numpy arrays of rank at least 2, while the
    RECEIVERS, SENDERS are numpy arrays of rank 1 and same dimension as the
    EDGES field first dimension. The GLOBALS field is a numpy array of rank at
    least 1.

##### Returns:

  An instance of `graphs.GraphsTuple` containing numpy arrays. The
  `RECEIVERS`, `SENDERS`, `N_NODE` and `N_EDGE` fields are cast to `np.int32`
  type.


### [`utils_np.get_graph(input_graphs, index)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_np.py?l=518)<!-- utils_np.get_graph .code-reference -->

Indexes into a graph.

Given a `graphs.GraphsTuple` containing arrays and an index (either
an `int` or a `slice`), index into the nodes, edges and globals to extract the
graphs specified by the slice, and returns them into an another instance of a
`graphs.GraphsTuple` containing `Tensor`s.

##### Args:


* `input_graphs`: A `graphs.GraphsTuple` containing numpy arrays.
* `index`: An `int` or a `slice`, to index into `graph`. `index` should be
    compatible with the number of graphs in `graphs`.

##### Returns:

  A `graphs.GraphsTuple` containing numpy arrays, made of the extracted
    graph(s).

##### Raises:


* `TypeError`: if `index` is not an `int` or a `slice`.


### [`utils_np.graphs_tuple_to_data_dicts(graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_np.py?l=384)<!-- utils_np.graphs_tuple_to_data_dicts .code-reference -->

Splits the stored data into a list of individual data dicts.

Each list is a dictionary with fields NODES, EDGES, GLOBALS, RECEIVERS,
SENDERS.

##### Args:


* `graph`: A `graphs.GraphsTuple` instance containing numpy arrays.

##### Returns:

  A list of the graph data dictionaries. The GLOBALS field is a tensor of
    rank at least 1, as the RECEIVERS and SENDERS field (which have integer
    values). The NODES and EDGES fields have rank at least 2.


### [`utils_np.graphs_tuple_to_networkxs(graphs_tuple)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_np.py?l=343)<!-- utils_np.graphs_tuple_to_networkxs .code-reference -->

Converts a `graphs.GraphsTuple` to a sequence of networkx graphs.

##### Args:


* `graphs_tuple`: A `graphs.GraphsTuple` instance containing numpy arrays.

##### Returns:

  The list of `networkx.MultiDiGraph`s.


### [`utils_np.networkx_to_data_dict(graph_nx, node_shape_hint=None, edge_shape_hint=None, data_type_hint=float32)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_np.py?l=108)<!-- utils_np.networkx_to_data_dict .code-reference -->

Returns a data dict of Numpy data from a networkx graph.

The networkx graph should be set up such that, for fixed shapes `node_shape`,
 `edge_shape` and `global_shape`:
  - `graph_nx.nodes(data=True)[i][-1]["features"]` is, for any node index i, a
    tensor of shape `node_shape`, or `None`;
  - `graph_nx.edges(data=True)[i][-1]["features"]` is, for any edge index i, a
    tensor of shape `edge_shape`, or `None`;
  - `graph_nx.edges(data=True)[i][-1]["index"]`, if present, defines the order
    in which the edges will be sorted in the resulting `data_dict`;
  - `graph_nx.graph["features"] is a tensor of shape `global_shape`, or
    `None`.

The dictionary `type_hints` can provide hints of the "float" and "int" types
for missing values.

##### The output data is a sequence of data dicts with fields:

  NODES, EDGES, RECEIVERS, SENDERS, GLOBALS, N_NODE, N_EDGE.

##### Args:


* `graph_nx`: A `networkx.MultiDiGraph`.
* `node_shape_hint`: (iterable of `int` or `None`, default=`None`) If the graph
    does not contain nodes, the trailing shape for the created `NODES` field.
    If `None` (the default), this field is left `None`. This is not used if
    `graph_nx` contains at least one node.
* `edge_shape_hint`: (iterable of `int` or `None`, default=`None`) If the graph
    does not contain edges, the trailing shape for the created `EDGES` field.
    If `None` (the default), this field is left `None`. This is not used if
    `graph_nx` contains at least one edge.
* `data_type_hint`: (numpy dtype, default=`np.float32`) If the `NODES` or
    `EDGES` fields are autocompleted, their type.

##### Returns:

  The data `dict` of Numpy data.

##### Raises:


* `TypeError`: If `graph_nx` is not an instance of networkx.
* `KeyError`: If `graph_nx` contains at least one node without the "features"
    key in its attribute dictionary, or at least one edge without the
    "features" key in its attribute dictionary.
* `ValueError`: If `graph_nx` contains at least one node with a `None`
    "features" attribute and one least one node with a non-`None` "features"
    attribute; or if `graph_nx` contains at least one edge with a `None`
    "features" attribute and one least one edge with a non-`None` "features"
    attribute.


### [`utils_np.networkxs_to_graphs_tuple(graph_nxs, node_shape_hint=None, edge_shape_hint=None, data_type_hint=float32)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_np.py?l=289)<!-- utils_np.networkxs_to_graphs_tuple .code-reference -->

Constructs an instance from an iterable of networkx graphs.

 The networkx graph should be set up such that, for fixed shapes `node_shape`,
 `edge_shape` and `global_shape`:
  - `graph_nx.nodes(data=True)[i][-1]["features"]` is, for any node index i, a
    tensor of shape `node_shape`, or `None`;
  - `graph_nx.edges(data=True)[i][-1]["features"]` is, for any edge index i, a
    tensor of shape `edge_shape`, or `None`;
  - `graph_nx.edges(data=True)[i][-1]["index"]`, if present, defines the order
    in which the edges will be sorted in the resulting `data_dict`;
  - `graph_nx.graph["features"] is a tensor of shape `global_shape`, or
    `None`.

##### The output data is a sequence of data dicts with fields:

  NODES, EDGES, RECEIVERS, SENDERS, GLOBALS, N_NODE, N_EDGE.

##### Args:


* `graph_nxs`: A container of `networkx.MultiDiGraph`s.
* `node_shape_hint`: (iterable of `int` or `None`, default=`None`) If the graph
    does not contain nodes, the trailing shape for the created `NODES` field.
    If `None` (the default), this field is left `None`. This is not used if
    `graph_nx` contains at least one node.
* `edge_shape_hint`: (iterable of `int` or `None`, default=`None`) If the graph
    does not contain edges, the trailing shape for the created `EDGES` field.
    If `None` (the default), this field is left `None`. This is not used if
    `graph_nx` contains at least one edge.
* `data_type_hint`: (numpy dtype, default=`np.float32`) If the `NODES` or
    `EDGES` fields are autocompleted, their type.

##### Returns:

  The instance.

##### Raises:


* `ValueError`: If `graph_nxs` is not an iterable of networkx instances.


### [`utils_tf.concat(input_graphs, axis, name='graph_concat')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=350)<!-- utils_tf.concat .code-reference -->

Returns an op that concatenates graphs along a given axis.

In all cases, the NODES, EDGES and GLOBALS dimension are concatenated
along `axis` (if a fields is `None`, the concatenation is just a `None`).
If `axis` == 0, then the graphs are concatenated along the (underlying) batch
dimension, i.e. the RECEIVERS, SENDERS, N_NODE and N_EDGE fields of the tuples
are also concatenated together.
If `axis` != 0, then there is an underlying asumption that the receivers,
SENDERS, N_NODE and N_EDGE fields of the graphs in `values` should all match,
but this is not checked by this op.
The graphs in `input_graphs` should have the same set of keys for which the
corresponding fields is not `None`.

##### Args:


* `input_graphs`: A list of `graphs.GraphsTuple` objects containing `Tensor`s
    and satisfying the constraints outlined above.
* `axis`: An axis to concatenate on.
* `name`: (string, optional) A name for the operation.


* `Returns`: An op that returns the concatenated graphs.

##### Raises:


* `ValueError`: If `values` is an empty list, or if the fields which are `None`
    in `input_graphs` are not the same for all the graphs.


### [`utils_tf.data_dicts_to_graphs_tuple(data_dicts, name='data_dicts_to_graphs_tuple')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=924)<!-- utils_tf.data_dicts_to_graphs_tuple .code-reference -->

Creates a `graphs.GraphsTuple` containing tensors from data dicts.

 All dictionaries must have exactly the same set of keys with non-`None`
 values associated to them. Moreover, this set of this key must define a valid
 graph (i.e. if the `EDGES` are `None`, the `SENDERS` and `RECEIVERS` must be
 `None`, and `SENDERS` and `RECEIVERS` can only be `None` both at the same
 time). The values associated with a key must be convertible to `Tensor`s,
 for instance python lists, numpy arrays, or Tensorflow `Tensor`s.

 This method may perform a memory copy.

 The `RECEIVERS`, `SENDERS`, `N_NODE` and `N_EDGE` fields are cast to
 `np.int32` type.

##### Args:


* `data_dicts`: An iterable of data dictionaries with keys in `ALL_FIELDS`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A `graphs.GraphTuple` representing the graphs in `data_dicts`.


### [`utils_tf.fully_connect_graph_dynamic(graph, exclude_self_edges=False, name='fully_connect_graph_dynamic')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=768)<!-- utils_tf.fully_connect_graph_dynamic .code-reference -->

Adds edges to a graph by fully-connecting the nodes.

This method does not require the number of nodes per graph to be constant,
or to be known at graph building time.

##### Args:


* `graph`: A `graphs.GraphsTuple` with `None` values for the edges, senders and
    receivers.
  exclude_self_edges (default=False): Excludes self-connected edges.

* `name`: (string, optional) A name for the operation.

##### Returns:

  A `graphs.GraphsTuple` containing `Tensor`s with fully-connected edges.

##### Raises:


* `ValueError`: if any of the `EDGES`, `RECEIVERS` or `SENDERS` field is not
    `None` in `graph`.


### [`utils_tf.fully_connect_graph_static(graph, exclude_self_edges=False, name='fully_connect_graph_static')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=700)<!-- utils_tf.fully_connect_graph_static .code-reference -->

Adds edges to a graph by fully-connecting the nodes.

This method can be used if the number of nodes for each graph in `graph` is
constant and known at graph building time: it will be inferred by dividing
the number of nodes in the batch(the length of `graph.nodes`) by the number of
graphs in the batch (the length of `graph.n_node`). It is an error to call
this method with batches of graphs with dynamic or uneven sizes; in the latter
case, the method may silently yield an incorrect result.

##### Args:


* `graph`: A `graphs.GraphsTuple` with `None` values for the edges, senders and
    receivers.
  exclude_self_edges (default=False): Excludes self-connected edges.

* `name`: (string, optional) A name for the operation.

##### Returns:

  A `graphs.GraphsTuple` containing `Tensor`s with fully-connected edges.

##### Raises:


* `ValueError`: If any of the `EDGES`, `RECEIVERS` or `SENDERS` field is not
    `None` in `graph`.
* `ValueError`: If the number of graphs (extracted from `graph.n_node` leading
    dimension) or number of nodes (extracted from `graph.nodes` leading
    dimension) is not known at construction time, or if the latter does not
    divide the former (observe that this is only a necessary condition for
    the constantness of the number of nodes per graph).


### [`utils_tf.get_feed_dict(placeholders, graph)`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=217)<!-- utils_tf.get_feed_dict .code-reference -->

Feeds a `graphs.GraphsTuple` of numpy arrays or `None` into `placeholders`.

When feeding a fully defined graph (no `None` field) into a session, this
method is not necessary as one can directly do:

```
_ = sess.run(_, {placeholders: graph})
```

However, if the placeholders contain `None`, the above construction would
fail. This method allows to replace the above call by

```
_ = sess.run(_, get_feed_dict(placeholders: graph))
```

restoring the correct behavior.

##### Args:


* `placeholders`: A `graphs.GraphsTuple` containing placeholders.
* `graph`: A `graphs.GraphsTuple` containing placeholder compatibale values,
    or `None`s.

##### Returns:

  A dictionary with key placeholders and values the fed in values.

##### Raises:


* `ValueError`: If the `None` fields in placeholders and `graph` do not exactly
    match.


### [`utils_tf.get_graph(input_graphs, index, name='get_graph')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=955)<!-- utils_tf.get_graph .code-reference -->

Indexes into a graph.

Given a `graphs.graphsTuple` containing `Tensor`s and an index (either
an `int` or a `slice`), index into the nodes, edges and globals to extract the
graphs specified by the slice, and returns them into an another instance of a
`graphs.graphsTuple` containing `Tensor`s.

##### Args:


* `input_graphs`: A `graphs.GraphsTuple` containing `Tensor`s.
* `index`: An `int` or a `slice`, to index into `graph`. `index` should be
    compatible with the number of graphs in `graphs`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A `graphs.GraphsTuple` containing `Tensor`s, made of the extracted
    graph(s).

##### Raises:


* `TypeError`: if `index` is not an `int` or a `slice`.


### [`utils_tf.get_num_graphs(input_graphs, name='get_num_graphs')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=1023)<!-- utils_tf.get_num_graphs .code-reference -->

Returns the number of graphs (i.e. the batch size) in `input_graphs`.

##### Args:


* `input_graphs`: A `graphs.GraphsTuple` containing tensors.
* `name`: (string, optional) A name for the operation.

##### Returns:

  An `int` (if a static number of graphs is defined) or a `tf.Tensor` (if the
    number of graphs is dynamic).


### [`utils_tf.identity(graph, name='graph_identity')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=463)<!-- utils_tf.identity .code-reference -->

Pass each element of a graph through a `tf.identity`.

This allows, for instance, to push a name scope on the graph by writing:
```
with tf.name_scope("encoder"):
  graph = utils_tf.identity(graph)
```

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s. `None` values are passed
    through.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A `graphs.GraphsTuple` `graphs_output` such that for any field `x` in NODES,
  EDGES, GLOBALS, RECEIVERS, SENDERS, N_NODE, N_EDGE, if `graph.x` was
  `None`, `graph_output.x` is `None`, and otherwise
  `graph_output.x = tf.identity(graph.x)`


### [`utils_tf.make_runnable_in_session(graph, name='make_graph_runnable_in_session')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=488)<!-- utils_tf.make_runnable_in_session .code-reference -->

Allows a graph containing `None` fields to be run in a `tf.Session`.

The `None` values of `graph` are replaced by `tf.no_op()`. This function is
meant to be called just before a call to `sess.run` on a Tensorflow session
`sess`, as `None` values currently cannot be run through a session.

##### Args:


* `graph`: A `graphs.GraphsTuple` containing `Tensor`s or `None` values.
* `name`: (string, optional) A name for the operation.

##### Returns:

  A `graphs.GraphsTuple` `graph_output` such that, for any field `x` in NODES,
  EDGES, GLOBALS, RECEIVERS, SENDERS, N_NODE, N_EDGE, and a Tensorflow session
  `sess`, if `graph.x` was `None`, `sess.run(graph_output)` is `None`, and
  otherwise


### [`utils_tf.placeholders_from_data_dicts(data_dicts, force_dynamic_num_graphs=False, name='placeholders_from_data_dicts')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=261)<!-- utils_tf.placeholders_from_data_dicts .code-reference -->

Constructs placeholders compatible with a list of data dicts.

##### Args:


* `data_dicts`: An iterable of data dicts containing numpy arrays.
* `force_dynamic_num_graphs`: A `bool` that forces the batch dimension to be
    dynamic. Defaults to `True`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  An instance of `graphs.GraphTuple` placeholders compatible with the
    dimensions of the dictionaries in `data_dicts`.


### [`utils_tf.placeholders_from_networkxs(graph_nxs, node_shape_hint=None, edge_shape_hint=None, data_type_hint=tf.float32, force_dynamic_num_graphs=True, name='placeholders_from_networkxs')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=282)<!-- utils_tf.placeholders_from_networkxs .code-reference -->

Constructs placeholders compatible with a list of networkx instances.

Given a list of networkxs instances, constructs placeholders compatible with
the shape of those graphs.

The networkx graph should be set up such that, for fixed shapes `node_shape`,
 `edge_shape` and `global_shape`:
  - `graph_nx.nodes(data=True)[i][-1]["features"]` is, for any node index i, a
    tensor of shape `node_shape`, or `None`;
  - `graph_nx.edges(data=True)[i][-1]["features"]` is, for any edge index i, a
    tensor of shape `edge_shape`, or `None`;
  - `graph_nx.edges(data=True)[i][-1]["index"]`, if present, defines the order
    in which the edges will be sorted in the resulting `data_dict`;
  - `graph_nx.graph["features"] is a tensor of shape `global_shape` or `None`.

##### Args:


* `graph_nxs`: A container of `networkx.MultiDiGraph`s.
* `node_shape_hint`: (iterable of `int` or `None`, default=`None`) If the graph
    does not contain nodes, the trailing shape for the created `NODES` field.
    If `None` (the default), this field is left `None`. This is not used if
    `graph_nx` contains at least one node.
* `edge_shape_hint`: (iterable of `int` or `None`, default=`None`) If the graph
    does not contain edges, the trailing shape for the created `EDGES` field.
    If `None` (the default), this field is left `None`. This is not used if
    `graph_nx` contains at least one edge.
* `data_type_hint`: (numpy dtype, default=`np.float32`) If the `NODES` or
    `EDGES` fields are autocompleted, their type.
* `force_dynamic_num_graphs`: A `bool` that forces the batch dimension to be
    dynamic. Defaults to `True`.
* `name`: (string, optional) A name for the operation.

##### Returns:

  An instance of `graphs.GraphTuple` placeholders compatible with the
    dimensions of the graph_nxs.


### [`utils_tf.repeat(tensor, repeats, axis=0, name='repeat')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=510)<!-- utils_tf.repeat .code-reference -->

Repeats a `tf.Tensor`'s elements along an axis by custom amounts.

Equivalent to Numpy's `np.repeat`.
`tensor and `repeats` must have the same numbers of elements along `axis`.

##### Args:


* `tensor`: A `tf.Tensor` to repeat.
* `repeats`: A 1D sequence of the number of repeats per element.
* `axis`: An axis to repeat along. Defaults to 0.
* `name`: (string, optional) A name for the operation.

##### Returns:

  The `tf.Tensor` with repeated values.


### [`utils_tf.set_zero_edge_features(graph, edge_size, dtype=tf.float32, name='set_zero_edge_features')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=857)<!-- utils_tf.set_zero_edge_features .code-reference -->

Completes the edge state of a graph.

##### Args:


* `graph`: A `graphs.GraphsTuple` with a `None` edge state.
* `edge_size`: (int) the dimension for the created edge features.
* `dtype`: (tensorflow type) the type for the created edge features.
* `name`: (string, optional) A name for the operation.

##### Returns:

  The same graph but for the edge field, which is a `Tensor` of shape
  `[number_of_edges, edge_size]`, where `number_of_edges = sum(graph.n_edge)`,
  with type `dtype` and filled with zeros.

##### Raises:


* `ValueError`: If the `EDGES` field is not None in `graph`.
* `ValueError`: If the `RECEIVERS` or `SENDERS` field are None in `graph`.
* `ValueError`: If `edge_size` is None.


### [`utils_tf.set_zero_global_features(graph, global_size, dtype=tf.float32, name='set_zero_global_features')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=893)<!-- utils_tf.set_zero_global_features .code-reference -->

Completes the global state of a graph.

##### Args:


* `graph`: A `graphs.GraphsTuple` with a `None` global state.
* `global_size`: (int) the dimension for the created global features.
* `dtype`: (tensorflow type) the type for the created global features.
* `name`: (string, optional) A name for the operation.

##### Returns:

  The same graph but for the global field, which is a `Tensor` of shape
  `[num_graphs, global_size]`, type `dtype` and filled with zeros.

##### Raises:


* `ValueError`: If the `GLOBALS` field of `graph` is not `None`.
* `ValueError`: If `global_size` is not `None`.


### [`utils_tf.set_zero_node_features(graph, node_size, dtype=tf.float32, name='set_zero_node_features')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=825)<!-- utils_tf.set_zero_node_features .code-reference -->

Completes the node state of a graph.

##### Args:


* `graph`: A `graphs.GraphsTuple` with a `None` edge state.
* `node_size`: (int) the dimension for the created node features.
* `dtype`: (tensorflow type) the type for the created nodes features.
* `name`: (string, optional) A name for the operation.

##### Returns:

  The same graph but for the node field, which is a `Tensor` of shape
  `[number_of_nodes, node_size]`  where `number_of_nodes = sum(graph.n_node)`,
  with type `dtype`, filled with zeros.

##### Raises:


* `ValueError`: If the `NODES` field is not None in `graph`.
* `ValueError`: If `node_size` is None.


### [`utils_tf.stop_gradient(graph, stop_edges=True, stop_nodes=True, stop_globals=True, name='graph_stop_gradient')`](https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py?l=418)<!-- utils_tf.stop_gradient .code-reference -->

Stops the gradient flow through a graph.

##### Args:


* `graph`: An instance of `graphs.GraphsTuple` containing `Tensor`s.
* `stop_edges`: (bool, default=True) indicates whether to stop gradients for
    the edges.
* `stop_nodes`: (bool, default=True) indicates whether to stop gradients for
    the nodes.
* `stop_globals`: (bool, default=True) indicates whether to stop gradients for
    the globals.
* `name`: (string, optional) A name for the operation.

##### Returns:

  GraphsTuple after stopping the gradients according to the provided
  parameters.

##### Raises:


* `ValueError`: If attempting to stop gradients through a field which has a
    `None` value in `graph`.


