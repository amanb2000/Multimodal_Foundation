"""
Prototype: Asynchronously prefetching generator.

This is a very small worked example of a pythonic generator that performs item 
retrieval in the background for future requests (i.e., prefetches) from an I/O-
bound routine. 

For simplicity, the I/O-bound operation will be a `time.sleep()` operation. 
Similar for the consumer's "work" function.  I will also be checking PID's to 
ensure that it is ACTUALLY a new process performing the work. 

This leverages two objects from `multiprocessing`: a Queue and the Process 
object. The Queue is process-safe and enables blocking until data is received 
via the `Queue.get()` function (waits until queue size is greater than 0). 


""" 

import time
import os
import pdb
import multiprocessing
import random

import numpy as np

## Fake data retriever function. 
def extend_queue(queue, wait_time, cnt): 
	""" 
	Simulates an IO-heavy process that requries `wait_time` seconds to push 
	a piece of data to some `queue`. 
	""" 
	print(f"\tPID {os.getpid()} sleeping now...")
	new_val = cnt # = random.randint(0,33)
	print("\tNew value: ", new_val)
	time.sleep(wait_time)
	queue.put(new_val)

def get_and_replenish(queue, wait_time,cnt=3):
	""" Gets a piece of data after launching a process to replenish.
	"""
	worker = multiprocessing.Process(target=extend_queue, args=(queue, wait_time,cnt))
	worker.start()
	return queue.get()

## Generator Creation -- fixed IO time.
def get_generator(data_getter, IO_time=1.0, cnt=1): 
	io_time = IO_time

	def generate_data():
		# Actual generator class. 
		_io_time = io_time
		_data_getter = data_getter

		data_cache = multiprocessing.Queue()
		worker = multiprocessing.Process(target=extend_queue, args=(data_cache, _io_time))
		worker.start()
		worker.join()
		cnt=1
		_data_getter(data_cache, _io_time, cnt=cnt)
		cnt += 1
		while True: 
			yield _data_getter(data_cache, _io_time,cnt=cnt)
			cnt+=1

			# yield _data_getter(_io_time)

	return generate_data

## Consumer Creation -- fixed work time.

## Test: No pre-fetching

## Test: With pre-fetching


if __name__ == "__main__":
	print("Parent process PID: ", os.getpid())
	generator = get_generator(get_and_replenish, IO_time=1.7)
	# pdb.set_trace()

	cnt = 0
	start = time.time()
	for i in generator():
		cnt += 1
		time.sleep(1.5) # use data
		end = time.time()
		print("Datum: ", i)
		print("Cycle interval: ", end-start)
		
		start = time.time()
		if cnt == 30: 
			break