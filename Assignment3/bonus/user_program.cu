﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  for (int i = 0; i < input_size; i++)
    vm_write(vm, i, input[i]);

  for (int i = input_size - 1; i >= input_size - 32769; i--)
    int value = vm_read(vm, i);

  vm_snapshot(vm, results, 0, input_size);
}

/*
__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
  int input_size) {
// write the data.bin to the VM starting from address 32*1024
for (int i = 0; i < input_size; i++)
  vm_write(vm, 32*1024+i, input[i]);
  printf("step1:%d\n",*(vm->pagefault_num_ptr));
// write (32KB-32B) data  to the VM starting from 0
for (int i = 0; i < 32*1023; i++)
  vm_write(vm, i, input[i+32*1024]);
  printf("step2:%d\n",*(vm->pagefault_num_ptr)-4096);
// readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
vm_snapshot(vm, results, 32*1024, input_size);
printf("step3:%d\n",*(vm->pagefault_num_ptr)-5119);
}
*/
// expected page fault num: 9215
/*
__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  printf("swap_table:%d\n",vm->swap_table[1]);
  for (int i = 0; i < 32*1024; i++)
    vm_write(vm, i, input[i]);
  vm_write(vm,32*1024,input[32*1024]);
  vm_write(vm,0,input[32*1024+1]);
  //int value = vm_read(vm,0);
  //vm_snapshot(vm, results, 0, input_size);
                             }
*/