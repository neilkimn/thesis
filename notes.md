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