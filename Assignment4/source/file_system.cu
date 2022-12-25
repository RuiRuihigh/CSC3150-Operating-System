#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 file_number = 0;
__device__ __managed__ u32 com_time = 0;
__device__ void volume_init(FileSystem *fs){
  for(int i=0;i<4096;i++){
    fs->volume[i]=(uchar)0Xff;
  }
  for(int i=4096;i<1085440;i++){
    fs->volume[i]=(uchar)0X00;
  }
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
  volume_init(fs);
}


__device__ int occupied_block_num(FileSystem *fs,int fcb_id){
  //int start_block = (fs->volume[4096+32*fcb_id+28]<<8)+fs->volume[4096+32*fcb_id+29];
  int this_size = (fs->volume[4096+32*fcb_id+24]<<24)+(fs->volume[4096+32*fcb_id+25]<<16)+(fs->volume[4096+32*fcb_id+26]<<8)+(fs->volume[4096+32*fcb_id+27]);
  int this_block_num;
  if(this_size%32==0){
    this_block_num = this_size/32;
  }    
  else{
    this_block_num = (this_size/32)+1;
  }
  return this_block_num;
}

__device__ int find_next_FCB(FileSystem *fs){
  int begin = 4096;
  for(int i=begin;i<=36832;i=i+32){
    if(fs->volume[i+31]==0x00){
      return i;
    }
  }
}

__device__ int search_file(FileSystem *fs, char *s){//return fcb id
  for(int i=0;i<1024;i++){
    int current = i*32+4096;
    if(fs->volume[current+31]==0x0f){
      int j = 0;
      int k = 0;
      while(s[j]!='\0' && fs->volume[current+k]!='\0' && s[j]==fs->volume[current+k]){
        j++;
        k++;
      }
      if(s[j]=='\0' && fs->volume[current+k]=='\0'){
        return i;
      }
    }
  }
  return -1;
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	if(op==G_READ){
    int check = search_file(fs,s);
    if(check==-1){
      printf("The file is not exist, cannot read!");
      return 0;
    }
    else{
      return 0xff000000+(u32)check;
    }
  }
  else{
    int check = search_file(fs,s);
    //printf("here%d\n",check);
    if(check!=-1){
      return 0x00ff0000+(u32)check;
    }
    else{
      int fcb_id = find_next_FCB(fs);//real index!!!
      fs->volume[fcb_id+31] = 0x0f;//used
      fs->volume[fcb_id+30] = 0x0f;//empty
      /*
      fs->volume[fcb_id+22] = gtime>>8;//modify time
      fs->volume[fcb_id+23] = gtime;
      fs->volume[fcb_id+20] = gtime>>8;//create time
      fs->volume[fcb_id+21] = gtime;
      */
      int j=0;
      while(s[j]!='\0'){
        fs->volume[fcb_id+j]=s[j];
        j++;
      }
      fs->volume[fcb_id+j]='\0';
      j=j+1;
      while(j<=19){
        fs->volume[fcb_id+j]=0x00;
        j++;
      }
      //gtime = gtime+1;
      int result = (fcb_id-4096)/32;
      return 0x00ff0000+(u32)result;
    }
  }
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	int level = fp>>16;
  if(level==0x00ff){
    printf("The open is for write,not allow!");
  }
  else{
    int fcbid = fp-0xff000000;
    int start_block1 = fs->volume[4096+fcbid*32+28]<<8;
    int start_block2 = fs->volume[4096+fcbid*32+29];
    int start_block = start_block1+start_block2;
    for(int i=0;i<size;i++){
      output[i]=fs->volume[start_block*32+fs->FILE_BASE_ADDRESS+i];
    }
  }
}

__device__ void clean_superblock(FileSystem *fs,int blockadd){
  int ind = blockadd/8;
  int remain = blockadd%8;
  uchar mask;
  mask = 1<<(8-(remain+1));
  //printf("other%d\n",fs->volume[ind]);
  //printf("mask%d\n",mask);
  
  fs->volume[ind] = fs->volume[ind] | mask;
  //printf("combine%d\n",fs->volume[ind]);
}

__device__ void clean_file(FileSystem *fs,int blockadd){
  int real_ind = fs->FILE_BASE_ADDRESS+blockadd*32;
  for (int k =0;k<=31;k++){
    fs->volume[real_ind+k] = 0;
  }
}

__device__ void set_superblock(FileSystem *fs,int blockadd){
  int ind = blockadd/8;
  int remain = blockadd%8;
  uchar mask;
  mask = 1<<(8-(remain+1));
  mask = 0b11111111-mask;
  fs->volume[ind] = fs->volume[ind] & mask;
}

__device__ u32 remain_bit(FileSystem *fs){//vcb中有几个bit剩余
  u32 free_block_num = 0;
  for(int i=4095;i>=0;i--){
    if(fs->volume[i]==0xff){
      free_block_num += 8;
    }
    else{
      for(int j=0;j<=7;j++){
        if((fs->volume[i] & (1<<j))-(1<<j)==0){//fs->volume[i] & (1<<j)==(1<<j)
          free_block_num += 1;
        }
        else{
          break;
        }
      }
      break;
    }
  }
  return free_block_num;
}

__device__ u32 find_fcb(FileSystem *fs,u32 block_id){//用startblock找fcb——id
  for(int i=0;i<1024;i++){
    u32 current = 4096+i*32;
    if(fs->volume[current+31]==0x0f && fs->volume[current+30]==0xff){//fs->volume[current+31]==0x0f && fs->volume[current+30]==0xff
      u32 temp = (fs->volume[current+28]<<8) + fs->volume[current+29];
      if(temp==block_id){
        return i;
      }
    }
  }
  return 0xffffffff;
}

__device__ u32 find_first_1(FileSystem *fs,u32 startplace){
  u32 ind = startplace/8;
  u32 remain = startplace%8;
  //printf("%d\n",ind);
  //printf("%d\n",remain);
  for(u32 i = ind;i<4096;i++){
    if(i==ind){
      uchar temp = fs->volume[i];
      //printf("%d\n",temp);
      for(u32 j=remain;j<=7;j++){
        uchar mask = 1<<(7-j);
        //printf("%d\n",mask-(temp&mask)==0);
        if(mask-(temp&mask)==0 ){//mask == temp&mask 
          return i*8+j;
        }
      }
    }
    else{
      uchar temp = fs->volume[i];
      for(u32 j=0;j<=7;j++){
        uchar mask = 1<<(7-j);
        if(mask-(temp&mask)==0 ){//mask == temp&mask
          return i*8+j;
        }
      }
    }
  }
  return 0xffff0000;
}

__device__ u32 find_first_0(FileSystem *fs,u32 startplace){
  u32 ind = startplace/8;
  u32 remain = startplace%8;
  for(u32 i=ind;i<4096;i++){
    if(i==ind){
      uchar temp = fs->volume[i];
      for(u32 j=remain;j<=7;j++){
        uchar mask = 1<<(7-j);
        mask = 0b11111111 - mask;
        //printf("%d\n",mask-(temp|mask)==0);
        if(mask-(temp|mask)==0){//mask== temp|mask
          return i*8+j;
        }
      }
    }
    else{
      uchar temp = fs->volume[i];
      for(u32 j=0;j<=7;j++){
        uchar mask = 1<<(7-j);
        mask = 0b11111111 - mask;
        if(mask-(temp|mask)==0){//mask== temp|mask
          return i*8+j;
        }
      }
    }
  }
  return 0xffff;
}

__device__ void move(FileSystem *fs,u32 from_block,u32 to_block){
  for(int i=0;i<=31;i++){
    fs->volume[fs->FILE_BASE_ADDRESS+to_block*32+i] = fs->volume[fs->FILE_BASE_ADDRESS+from_block*32+i];
  }
}

__device__ void compact(FileSystem *fs,u32 start_place){
  //com_time ++;
  //printf("I am compacting!!!");
  //printf("here%d\n",fs->volume[0]);
  u32 temp1 = find_first_1(fs,start_place);//pt1
  //printf("%d\n",temp1);
  if(temp1!=0xffff0000){
    u32 temp2 = find_first_0(fs,temp1+1);//pt2
    //printf("%d\n",temp2);
    if(temp2!=0xffff){
      u32 temp3 = find_fcb(fs,temp2);//fcb_id
      //printf("%d\n",temp3);
      //update fcb
      fs->volume[4096+temp3*32+28] = temp1>>8;
      fs->volume[4096+temp3*32+29] = temp1;
      /*
      u32 this_size = fs->volume[4096+temp3*32+24]<<24+fs->volume[4096+temp3*32+25]<<16+fs->volume[4096+temp3*32+26]<<8+fs->volume[4096+temp3*32+27];
      u32 this_block_num;
      if(this_size%32==0){
        this_block_num = this_size/32;
      }
      else{
        this_block_num = (this_size/32)+1;
      }
      */
      u32 this_block_num = occupied_block_num(fs,temp3);
      //printf("%d\n",this_block_num);
      //printf("\n");
      for(u32 i=0;i<this_block_num;i++){
        move(fs,temp2,temp1);//swap file
        //update vcb
        clean_superblock(fs,temp2);
        set_superblock(fs,temp1);
        temp2 +=1;
        temp1 +=1;
      }
      //printf("compacting\n");
      compact(fs,temp1);
    }
  }
  else{
    printf("The volume is totaly full");
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	int level = fp>>16;
  if(level==0xff00){
    printf("The open is for read,not allow!");
  }
  else{
    int fcbid = fp-0x00ff0000;
    int empty_sta = fs->volume[4096+fcbid*32+30];
    if(empty_sta==0xff){//overwrite
      fs->volume[4096+fcbid*32+22] = gtime>>8;//modify time
      fs->volume[4096+fcbid*32+23] = gtime;
      gtime +=1;

      //int start_block1 = fs->volume[4096+fcbid*32+28]<<8;
      //int start_block2 = fs->volume[4096+fcbid*32+29];
      
      int start_block = (fs->volume[4096+fcbid*32+28]<<8) + fs->volume[4096+fcbid*32+29];
      //printf("%d\n",start_block);
      /*
      int occupied_size = fs->volume[4096+fcbid*32+24]<<24 + fs->volume[4096+fcbid*32+25]<<16 + fs->volume[4096+fcbid*32+26]<<8 + fs->volume[4096+fcbid*32+27];
      int occupied_blocks;
      if(occupied_size%32==0){
        occupied_blocks = (occupied_size/32);
      }
      else{
        occupied_blocks = (occupied_size/32)+1;
      }
      */
      //int occupied_blocks = (occupied_size/32)+1;//no need to plus 1?
      u32 occupied_blocks = occupied_block_num(fs,fcbid);
      //printf("%d\n",occupied_blocks);
      for(int i=0;i<occupied_blocks;i++){
        clean_superblock(fs,start_block+i);
        //clean_file(fs,start_block+i);
      }
      //printf("this?:%d\n",fs->volume[0]);
    }//上次找到这！！！！！！！！！！！
    else{//new write
      fs->volume[4096+fcbid*32+22] = gtime>>8;//modify time
      fs->volume[4096+fcbid*32+23] = gtime;
      fs->volume[4096+fcbid*32+20] = gtime>>8;//create time
      fs->volume[4096+fcbid*32+21] = gtime;
      gtime +=1;
      file_number +=1;
      //fs->volume[4096+fcbid*32+30]=0xff;
    }

    //判断size够不够
    u32 remainbit = remain_bit(fs);
    //printf("%d\n",remainbit);
    u32 remain_size = remainbit*32;
    if(size<=remain_size){
      u32 load_block_start = (1<<15)-remainbit;
      fs->volume[4096+fcbid*32+28] = load_block_start>>8;
      fs->volume[4096+fcbid*32+29] = load_block_start;
      fs->volume[4096+fcbid*32+30] = 0xff;
      fs->volume[4096+fcbid*32+24] = size>>24;
      fs->volume[4096+fcbid*32+25] = size>>16;
      fs->volume[4096+fcbid*32+26] = size>>8;
      fs->volume[4096+fcbid*32+27] = size;
      //write
      //int temp = -1;
      for (int i=0;i<size;i++){
        /* update the disk */
        fs->volume[fs->FILE_BASE_ADDRESS + load_block_start * 32 + i] = input[i];
        int change = i/32;
        //if(temp<change){
        //  temp = change;
        set_superblock(fs,load_block_start+change);
        //}
      }
      //printf("then?%d\n",fs->volume[0]);
      //printf("%d\n",remain_bit(fs));
    }
    else{
      compact(fs,0);
      //printf("now here");
      com_time ++;
      u32 remainbit1 = remain_bit(fs);
      //printf("%d\n",remainbit1);
      if(size<=remainbit1*32){
        u32 load_block_start = (1<<15)-remainbit1;
        fs->volume[4096+fcbid*32+28] = load_block_start>>8;
        fs->volume[4096+fcbid*32+29] = load_block_start;
        fs->volume[4096+fcbid*32+30] = 0xff;
        fs->volume[4096+fcbid*32+24] = size>>24;
        fs->volume[4096+fcbid*32+25] = size>>16;
        fs->volume[4096+fcbid*32+26] = size>>8;
        fs->volume[4096+fcbid*32+27] = size;
        //write
        //int temp = -1;
        for (int i=0;i<size;i++){
          /* update the disk */
          fs->volume[fs->FILE_BASE_ADDRESS + load_block_start * 32 + i] = input[i];
          int change = i/32;
          //if(temp<change){
          //  temp = change;
          set_superblock(fs,load_block_start+change);
          //}
        }
      }
      else{
        printf("Not enough size for the file!!");
      }
    }

  }
}
/*
__device__ void print_name(FileSystem *fs,u32 fcb_id){
  char result[20];
  for(int i=0;i<20;i++){
    result[i] = fs->volume[4096+32*fcb_id+i];
  }
  printf("%d\n",result);
}

__device__ void print_name_size(FileSystem *fs,u32 fcb_id,int size){
  char result[20];
  for(int i=0;i<20;i++){
    result[i] = fs->volume[4096+32*fcb_id+i];
  }
  printf("%d ",result);
  printf("%d\n",size);
}
*/
__device__ void print_string(uchar *s){
  char temp[20];
  int i = 0;
  while(*s!='\0'){
    //printf("%c", (char) *s);
    temp[i] = (char) *s;
    i++;
    s = s+1;
  }
  temp[i] = '\0';
  printf("%s",temp);
}


__device__ void fs_gsys(FileSystem *fs, int op)
{
  if(op==LS_D){//list by modify time
    printf("===sort by modified time===\n");
    int time = 0;
    int ub = (1<<30);
    while (time<file_number){
      int max=-1;//0
      int position = 0;
      for(int i=0;i<1024;i++){
        if(fs->volume[4096+32*i+31]==0x0f){
          int mod_time = (fs->volume[4096+32*i+22]<<8)+fs->volume[4096+32*i+23];
          if(max<mod_time && mod_time<ub){
            max = mod_time;
            position = i;
          }
        }
      }
      //printf("%d\n",time);
      uchar* name = &fs->volume[4096+32*position];
      print_string(name);
      printf("\n");
      time = time+1;
      ub = max;
    }
    //printf("%d\n",com_time);
  }
  else if(op==LS_S){
    printf("===sort by file size===\n");
    int time = 0;
    bool counted[1024];
    for(int j=0;j<1024;j++){
      counted[j] = false;
    }
    while(time<file_number){
      int min_createtime = (1<<30);
      int max_size = 0;
      int position = 1026;
      for(int i=0;i<1024;i++){
        if(fs->volume[4096+32*i+31]==0x0f){
          int this_size = (fs->volume[4096+32*i+24]<<24)+(fs->volume[4096+32*i+25]<<16)+(fs->volume[4096+32*i+26]<<8)+(fs->volume[4096+32*i+27]);
          int this_create_time = (fs->volume[4096+32*i+20]<<8)+fs->volume[4096+32*i+21];
          if(this_size>max_size && counted[i]==false){
            max_size = this_size;
            min_createtime = this_create_time;
            position = i;
          }
          else if(this_size==max_size && this_create_time<min_createtime && counted[i]==false){
            //max_size = this_size;
            min_createtime = this_create_time;
            position = i;
          }
        }
      }
      if(position==1026){
        time = time+1;
      }
      else{
        uchar* name = &fs->volume[4096+32*position];
        print_string(name);
        printf(" %d\n",max_size);
        //print_name_size(fs,position,max_size);
        time = time+1;
        counted[position] = true;
      }
    }
  }
}

__device__ void clean_fcb_entry(FileSystem *fs,int fcb_id){
  for(int i=0;i<32;i++){
    fs->volume[4096+fcb_id*32+i] = 0x00;
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	if(op==RM){
    int temp = search_file(fs,s);
    if(temp==-1){
      printf("No such file name,cannot be deleted!");
    }
    else{
      file_number = file_number-1;
      int start_block = (fs->volume[4096+32*temp+28]<<8)+fs->volume[4096+32*temp+29];
      int occupied_blocks = occupied_block_num(fs,temp);
      for(int i=0;i<occupied_blocks;i++){
        clean_superblock(fs,start_block+i);
      }
      clean_fcb_entry(fs,temp);
    }
  }
}




