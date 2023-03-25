Multiple training

1 train, batch size 30
2041 MB
777 MB

Method: Sharing data
2 training instances, batch size 30
577 MB - parent process (50MB memory allocated to tensors)
2041 MB - child process 1
2041 MB - child process 2

Method: Single training
2 training instances, batch size 30
2097 MB - process 1
2097 MB - process 2


3 train, batch size 30
2041 MB
2041 MB
2041 MB
651 MB

1 train, batch size 30, queue depth 1
2041 MB
615 MB

1 train, batch size 30, queue depth 1, pinned mem disabled
2041 MB
595 MB

2 train, batch size 80
3373 MB
3373 MB
679 MB
total: 7425 MB

Dummy data makes no difference (since dummy data generated on CPU side)

Single training

1 train, batch size 30
2097 MB

2 train, batch size 30
2097 MB
2097 MB

etc., scales

2 train, batch size 80
3701 MB
3701 MB
total: 7402 MB

2 train, batch size 100
3577 MB
3793 MB



### Notes 16 Mar

Single training, rn18 and rn34, 10 epochs, batch size 50
rn18: 2695MB
rn34: 3241MB

Multiple training, rn18 (40%) and rn34 (60%), 10 epochs, batch size 50
rn18: 2337MB
rn34: 2917MB
loader: 621MB
MPS: 23MB

Multiple training, rn18 (50%) and rn34 (50%), 10 epochs, batch size 50
rn18: 2403MB
rn34: 2957MB
loader: 631MB

# TODO
- Come up with measure of throttling for when using 50-50 split of GPU resources.

- Grid-search something-something to find 'sweet spot' for identical performance of RN18 and RN34.

- Narrow down CUDA context overhead.

- Compare 50-50 MIG vs default: No different in shared-dataset scenario

# TODO new
- Eval individual runs of RN18 and RN34 against the combined workloads. Also do RN18-RN18, RN34-RN34

- Figure out why memory consumption is low all of a sudden. Is it due to move from Rebelrig to own machine? Has to do with subclassing of Queue

- Update graphs with validation accuracy to be scatter

- 'Intelligent' MPS partitioning based on weights
