#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0;
  }
}
/*
__device__ void init_swap_table(VirtualMemory *vm){
  for (int i=0;i< 4*1024;i++){
    vm->swap_table[i]=0x80000000; // invalid := MSB is 1
  }
}
*/
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES, u32 *swap_table) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
  vm->swap_table = swap_table;
  //init_swap_table(vm);
}
/*
__device__ void show_size(VirtualMemory *vm){
  printf("size of swap:%d\n",sizeof(vm->swap_table));
}
*/
__device__ int invalid_bit_status(VirtualMemory *vm,int i){  //1 is empty; 0 is sth there
  if(vm->invert_page_table[i] == 0x80000000){
    return 1;
  }
  else{
    return 0;
  }
}
/*
__device__ u32 mask_pid(u32 table_entry){
  u32 temp = table_entry;
  temp = temp>>13;
  temp = temp&0b11;
  return temp;
}
__device__ u32 mask_pageNum(u32 table_entry){
  u32 temp = table_entry&0b1111_1111_1111_1;
  return temp;
}
*/
__device__ int page_table_status(VirtualMemory *vm){
  for (int k=0;k<vm->PAGE_ENTRIES; k++){
    if(invalid_bit_status(vm,k)==1){
        return k;//still have place
      }
    }
  return -1;//full
  }
__device__ int swap_table_status(VirtualMemory *vm){
  for (int k=0;k<4*1024; k++){
    if(vm->swap_table[k]==0x80000000){
        return k;//still have place
      }
    }
  return -2;//full
  }
__device__ int search_page_table(VirtualMemory *vm, u32 PageNum){
  for (int k=0;k<vm->PAGE_ENTRIES; k++){
    if(invalid_bit_status(vm,k)==0){
        if(vm->invert_page_table[k]==PageNum){
          return k;
        }
      }
    }
  return -3;//can not find
  }
__device__ int search_swap_table(VirtualMemory *vm, u32 PageNum){
  for (int k=0;k<4*1024; k++){
    if(vm->swap_table[k]!=0x80000000){
        if(vm->swap_table[k]==PageNum){
          return k;
        }
      }
    }
  return -4;//can not find
  }

__device__ void LRU_update(VirtualMemory *vm,int FrameNum){
  for (int j=0; j<vm->PAGE_ENTRIES; j++){
    if(invalid_bit_status(vm,j)==0){
      vm->invert_page_table[j + vm->PAGE_ENTRIES] = vm->invert_page_table[j + vm->PAGE_ENTRIES]+(u32)0x1;
    }
  }
  vm->invert_page_table[FrameNum+vm->PAGE_ENTRIES] = (u32)0x0;
}

__device__ int find_oldest_entry(VirtualMemory *vm){
  u32 max=0;
  int max_index;
  for (int j=0; j<vm->PAGE_ENTRIES; j++){
    if(invalid_bit_status(vm,j)==0){
      if(vm->invert_page_table[j + vm->PAGE_ENTRIES]>max){
        max = vm->invert_page_table[j + vm->PAGE_ENTRIES];
        max_index = j;
      }
    } 
  }
  return max_index;
}


__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  
  u32 offset = addr%32;
	u32 page_num = addr/32;
  uchar readout;
  if(search_page_table(vm,page_num)!=-3){//this pagenum on page table
    int frame_num;
    frame_num = search_page_table(vm,page_num);
    readout = vm->buffer[frame_num*32+offset];
    LRU_update(vm,frame_num);
    //printf("read from page table\n");
  }
  else{
    if(search_swap_table(vm,page_num)==-4){ //pagnum neither on page table nor on swap table
      printf("error:Can not find the content to read out!\n");
    }
    else{//pagnum not on page table but on swap table
      int j = search_swap_table(vm,page_num);//找到存有该page number的swap table的index
      int swap_index = find_oldest_entry(vm);//找出page table要被替换的entry的index
      int k = swap_table_status(vm);//找出second storage空的地方
      //开始做swap
      for(int i = 0; i < 32; i++){
        vm->storage[32*k+i] = vm->buffer[32*swap_index+i];
      }
      vm->swap_table[k] = vm->invert_page_table[swap_index];//替换page num 多pid要改

      for(int i = 0; i < 32; i++){
        vm->buffer[32*swap_index+i] = vm->storage[32*j+i];
      }
      vm->invert_page_table[swap_index] = vm->swap_table[j];//替换page num 多pid要改
      vm->swap_table[j] = 0x80000000;
      readout = vm->buffer[swap_index*32+offset];
      *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr)+1;
      LRU_update(vm,swap_index);
      //结束swap
    }
  }
  return readout; //TODO
  
}


__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  
  
  u32 offset = addr%32;
	u32 page_num = addr/32;
	int frame_num;
  if(search_page_table(vm,page_num)!=-3){//hit
    frame_num = search_page_table(vm,page_num);
    vm->buffer[frame_num*32+offset] = value;
    LRU_update(vm,frame_num);
    //printf("write hit\n");
  }
  else{//PN not on page table
    if(page_table_status(vm)!=-1){//page table not full
      frame_num = page_table_status(vm);
      vm->buffer[frame_num*32+offset] = value;
      vm->invert_page_table[frame_num] = page_num;
      LRU_update(vm,frame_num);
      *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr)+1;
      //printf("write new page table:%d\n",frame_num);
    }
    else{//page table full now 
      //printf("this:-1\n");
      if(search_swap_table(vm,page_num)==-4){//PN  not on swap table
        int swap_index = find_oldest_entry(vm);//找出page table要被替换的entry的index
        int k = swap_table_status(vm);//找出second storage空的地方
        //printf("ind%d\n",k);
        //printf("swap_index%d\n",swap_index);

        for(int i = 0; i < 32; i++){
          vm->storage[k*32+i] = vm->buffer[swap_index*32+i];
        }
        vm->swap_table[k] = vm->invert_page_table[swap_index];
        vm->invert_page_table[swap_index] = page_num;
        vm->buffer[swap_index*32+offset] = value;
        *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr)+1;
        LRU_update(vm,swap_index);
        //printf("write 3\n");
      }
      else{//PN  on swap table
        int j = search_swap_table(vm,page_num);//找到存有该page number的swap table的index
        int swap_index = find_oldest_entry(vm);//找出page table要被替换的entry的index
        int k = swap_table_status(vm);//找出second storage空的地方
        //printf("k%d\n",k);//1
        //printf("j%d\n",j);//0
        //printf("swap_index%d\n",swap_index);//1

        for(int i = 0; i < 32; i++){
          vm->storage[k*32+i] = vm->buffer[swap_index*32+i];
        }
        vm->swap_table[k] = vm->invert_page_table[swap_index];
        vm->invert_page_table[swap_index] = vm->swap_table[j];
        vm->swap_table[j] = 0x80000000;
        vm->buffer[swap_index*32+offset] = value;
        *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr)+1;
        LRU_update(vm,swap_index);
        printf("write 4\n");
      }
    }
  }
  
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */

   
  for (int j=offset;j<offset+input_size;j++){//input_size
    results[j-offset] = vm_read(vm,j);
  }
  
}

