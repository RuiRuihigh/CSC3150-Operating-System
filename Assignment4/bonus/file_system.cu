#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 file_number = 0;
__device__ __managed__ u32 com_time = 0;
__device__ __managed__ u32 current_folder = 0;


__device__ void set_root(FileSystem *fs){
  u32 root_fcbid = 0;
  current_folder = root_fcbid;
  fs->volume[4096+31] = 0xff;
  fs->volume[4096+30] = 0xff;
  fs->volume[4096+28] = 0>>8;
  fs->volume[4096+29] = 0;

  fs->volume[4096+24] = 0>>24;
  fs->volume[4096+25] = 0>>16;
  fs->volume[4096+26] = 0>>8;
  fs->volume[4096+27] = 0;

  fs->volume[36864] = 0;
  fs->volume[36864+1] = 0;
  fs->volume[36864+2] = 0;

  fs->volume[0] = 0x0f;
}


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
  set_root(fs);
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

__device__ u32 translate(FileSystem *fs,int i){
  u32 startblock = (fs->volume[4096+32*current_folder+28]<<8)+fs->volume[4096+32*current_folder+29];
  u32 fcbid = (fs->volume[36864+startblock*32+1+2*i]<<8)+fs->volume[36864+startblock*32+2+2*i];
  return fcbid;
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	if(op==G_READ){
    int check = search_file_in_current_folder(fs,s);
    if(check==-1){
      printf("The file is not exist, cannot read!");
      return 0;
    }
    else{
      u32 go = translate(fs,check);
      return 0xff000000+(u32)go;
    }
  }
  else{
    int check = search_file_in_current_folder(fs,s);
    //printf("here%d\n",check);
    if(check!=-1){
      u32 go = translate(fs,check);
      update_current_folder(fs,0,0,0);
      return 0x00ff0000+(u32)go;
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
      int result = (fcb_id-4096)/32;
      u32 child_f = (u32)result;
      u32 NS = check_name_size(s);
      update_current_folder(fs,NS,child_f,1);
      //gtime = gtime+1;
      //int result = (fcb_id-4096)/32;
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
    if(fs->volume[current+31]!=0){//fs->volume[current+31]==0x0f && fs->volume[current+30]==0xff
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
      if(fs->volume[4096+32*temp3+31]==0x0f){
        u32 this_block_num = occupied_block_num(fs,temp3);
        //printf("%d\n",this_block_num);
        //printf("\n");
        for(u32 i=0;i<this_block_num;i++){
          move(fs,temp2,temp1);//swap file
          //update vcb
          clean_superblock(fs,temp2);
          clean_file(fs,temp2);
          set_superblock(fs,temp1);
          temp2 +=1;
          temp1 +=1;
        }
      }
      else if(fs->volume[4096+32*temp3+31]==0xff){
        for(u32 i=0;i<4;i++){
          move(fs,temp2,temp1);//swap file
          //update vcb
          clean_superblock(fs,temp2);
          clean_file(fs,temp2);
          set_superblock(fs,temp1);
          temp2 +=1;
          temp1 +=1;
        }
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
        clean_file(fs,start_block+i);
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

/*
__device__ void print_string(uchar *s){
  while(*s!='\0'){
    printf("%c", (char) *s);
    s =s+1;
  }
}
*/

__device__ u32 current_folder_filenum(FileSystem *fs){
    u32 file_start_block = (fs->volume[4096+32*current_folder+28]<<8) + fs->volume[4096+32*current_folder+29];
    u32 result = fs->volume[36864+32*file_start_block];
    return result;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
  if(op==LS_D){//list by modify time
    printf("===sort by modified time===\n");
    int time = 0;
    int ub = (1<<30);
    u32 this_folder_filenum = current_folder_filenum(fs);
    int sub = (u32)this_folder_filenum;
    u32 file_start_block = (fs->volume[4096+32*current_folder+28]<<8) + fs->volume[4096+32*current_folder+29];
    u32 file_start_place = 36864+32*file_start_block;
    while (time<sub){
      int max=-1;
      int position = 0;
      for(int i=1;i<=50;i++){
        u32 the_fcbid = (fs->volume[file_start_place+1+2*i]<<8)+fs->volume[file_start_place+2+2*i];
        if(the_fcbid!=0){
          int mod_time = (fs->volume[4096+32*the_fcbid+22]<<8)+fs->volume[4096+32*the_fcbid+23];
          if(max<mod_time && mod_time<ub){
            //printf("%d\n",mod_time);
            max = mod_time;
            position = the_fcbid;
          }
        }
        /*
        if(fs->volume[4096+32*i+31]==0x0f){
          int mod_time = (fs->volume[4096+32*i+22]<<8)+fs->volume[4096+32*i+23];
          if(max<mod_time && mod_time<ub){
            max = mod_time;
            position = i;
          }
        }
        */
      }
      //printf("%d\n",time);
      uchar* name = &fs->volume[4096+32*position];
      if(fs->volume[4096+32*position+31]==0xff){
        print_string(name);
        printf(" d");
        printf("\n");
      }
      else if(fs->volume[4096+32*position+31]==0x0f){
        print_string(name);
        printf("\n");
      }
      time = time+1;
      ub = max;
    }
    //printf("%d\n",com_time);
  }
  else if(op==LS_S){
    printf("===sort by file size===\n");
    int time = 0;
    bool counted[50];
    for(int j=0;j<50;j++){
      counted[j] = false;
    }
    u32 this_folder_filenum = current_folder_filenum(fs);
    int sub = (u32)this_folder_filenum;
    u32 file_start_block = (fs->volume[4096+32*current_folder+28]<<8) + fs->volume[4096+32*current_folder+29];
    u32 file_start_place = 36864+32*file_start_block;
    while(time<sub){
      int ss=0;
      int min_createtime = (1<<30);
      int max_size = 0;
      int position = 1026;
      for(int i=0;i<50;i++){
        u32 the_fcbid = (fs->volume[file_start_place+3+2*i]<<8)+fs->volume[file_start_place+4+2*i];
        if(the_fcbid!=0){
          int this_size = (fs->volume[4096+32*the_fcbid+24]<<24)+(fs->volume[4096+32*the_fcbid+25]<<16)+(fs->volume[4096+32*the_fcbid+26]<<8)+(fs->volume[4096+32*the_fcbid+27]);
          int this_create_time = (fs->volume[4096+32*the_fcbid+20]<<8)+fs->volume[4096+32*the_fcbid+21];
          if(this_size>max_size && counted[i]==false){
            max_size = this_size;
            min_createtime = this_create_time;
            position = the_fcbid;
            ss = i;
          }
          else if(this_size==max_size && this_create_time<min_createtime && counted[i]==false){
            //max_size = this_size;
            min_createtime = this_create_time;
            position = the_fcbid;
            ss = i;
          }
        }
        
      /*
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
      */
      }
      if(position==1026){
        time = time+1;
      }
      else{
        uchar* name = &fs->volume[4096+32*position];
        if(fs->volume[4096+32*position+31]==0xff){
          print_string(name);
          printf(" %d",max_size);
          printf(" d");
          printf("\n");
        }
        else if(fs->volume[4096+32*position+31]==0x0f){
          print_string(name);
          printf(" %d",max_size);
          printf("\n");
        }
        //print_string(name);
        //printf(" %d\n",max_size);
        //print_name_size(fs,position,max_size);
        time = time+1;
        counted[ss] = true;
      }
    }
  }
  else if(op==PWD){
    u32 cf = current_folder;
    //printf("current%d\n",cf);
    print_path(fs,cf);
    //printf("\n");
  }
  else if(op==CD_P){
    u32 current_folder_file_start = (fs->volume[4096+32*current_folder+28]<<8)+fs->volume[4096+32*current_folder+29];
    u32 target_fcb = (fs->volume[36864+32*current_folder_file_start+1]<<8)+fs->volume[36864+32*current_folder_file_start+2];
    if(current_folder==0){
      printf("Already the root\n");
    }
    else{
      current_folder = target_fcb;
    }

  }
}

__device__ void clean_fcb_entry(FileSystem *fs,int fcb_id){
  for(int i=0;i<32;i++){
    fs->volume[4096+fcb_id*32+i] = 0x00;
  }
}

__device__ u32 check_name_size(char* s){
  u32 cou = 0;
  for(int i=0;i<20;i++){
    if(s[i]!='\0'){
      cou++;
    }
    else{
      break;
    }
  }
  cou = cou+1;
  return cou;
}

//mode 0:only update modify time 1:mkdir or open for write  2:RM
__device__ void update_current_folder(FileSystem *fs,u32 name_size,u32 child_fcbid,int mode){
  fs->volume[4096+32*current_folder+22] = gtime>>8;
  fs->volume[4096+32*current_folder+23] = gtime;
  if(parent(fs,current_folder)!=0){
    u32 parent_folder = parent(fs,current_folder);
    fs->volume[4096+32*parent_folder+22] = gtime>>8;
    fs->volume[4096+32*parent_folder+23] = gtime;
  }

  if(mode==1){
    //fs->volume[4096+32*current_folder+30] += name_size;   
    u32 temp_size = (fs->volume[4096+current_folder*32+24]<<24)+(fs->volume[4096+current_folder*32+25]<<16)+(fs->volume[4096+current_folder*32+26]<<8)+(fs->volume[4096+current_folder*32+27]);
    temp_size = temp_size + name_size;
    fs->volume[4096+current_folder*32+24] = temp_size>>24;
    fs->volume[4096+current_folder*32+25] = temp_size>>16;
    fs->volume[4096+current_folder*32+26] = temp_size>>8;
    fs->volume[4096+current_folder*32+27] = temp_size;


    u32 file_start_block = (fs->volume[4096+32*current_folder+28]<<8) + fs->volume[4096+32*current_folder+29];
    fs->volume[36864+32*file_start_block] += 1;
    //printf("current_folder_filenum:%d\n",fs->volume[36864+32*file_start_block]);
    //u32 temp = fs->volume[36864+32*file_start_block];
    for(u32 i=1;i<=50;i++){
      u32 this_fcbid = (fs->volume[36864+32*file_start_block+1+2*i]<<8)+fs->volume[36864+32*file_start_block+2+2*i];
      if(this_fcbid==0){
        fs->volume[36864+32*file_start_block+1+2*i] = child_fcbid>>8;
        fs->volume[36864+32*file_start_block+2+2*i] = child_fcbid;
        break;
      }
    }
    //fs->volume[36864+32*file_start_block+1+2*temp] = child_fcbid>>8;
    //fs->volume[36864+32*file_start_block+2+2*temp] = child_fcbid;
  }
  else if(mode==2){
    u32 temp_size = (fs->volume[4096+current_folder*32+24]<<24)+(fs->volume[4096+current_folder*32+25]<<16)+(fs->volume[4096+current_folder*32+26]<<8)+(fs->volume[4096+current_folder*32+27]);
    temp_size = temp_size - name_size;
    fs->volume[4096+current_folder*32+24] = temp_size>>24;
    fs->volume[4096+current_folder*32+25] = temp_size>>16;
    fs->volume[4096+current_folder*32+26] = temp_size>>8;
    fs->volume[4096+current_folder*32+27] = temp_size;

    u32 file_start_block = (fs->volume[4096+32*current_folder+28]<<8) + fs->volume[4096+32*current_folder+29];
    fs->volume[36864+32*file_start_block] -= 1;

    for(u32 i=1;i<=50;i++){
      u32 this_fcbid = (fs->volume[36864+32*file_start_block+1+2*i]<<8)+fs->volume[36864+32*file_start_block+2+2*i];
      if(this_fcbid==child_fcbid){
        fs->volume[36864+32*file_start_block+1+2*i] = 0>>8;
        fs->volume[36864+32*file_start_block+2+2*i] = 0;
        break;
      }
    }
  }
}

__device__ int search_file_in_current_folder(FileSystem *fs,char* s){//return i-th file
  u32 file_start_block = (fs->volume[current_folder*32+4096+28]<<8) + fs->volume[current_folder*32+4096+29];
  //printf("search_file_in_current_folde:%d\n",file_start_block);
  //u32 file_nums = fs->volume[36864+32*file_start_block];
  //printf("search_file_in_current_folde:%d\n",(fs->volume[36864+32*file_start_block+1+2*1]<<8)+fs->volume[36864+32*file_start_block+2+2*1]);
  //printf("search_file_in_current_folde:%d\n",(fs->volume[36864+32*file_start_block+1+2*2]<<8)+fs->volume[36864+32*file_start_block+2+2*2]);
  for(u32 i=1;i<=50;i++){
    u32 temp_fcbid = (fs->volume[36864+32*file_start_block+1+2*i]<<8)+fs->volume[36864+32*file_start_block+2+2*i];
    if(temp_fcbid!=0){
      u32 name_start = 4096+32*temp_fcbid;
      int j = 0;
      int k = 0;
      while(s[j]!='\0' && fs->volume[name_start+k]!='\0' && s[j]==fs->volume[name_start+k]){
        j++;
        k++;
      }
      if(s[j]=='\0' && fs->volume[name_start+k]=='\0'){
        //printf("i:%d\n",i);
        return i;
      }
    }
  }
  return -1;
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	if(op==RM){
    int temp = search_file_in_current_folder(fs,s);
    if(temp==-1){
      printf("No such file name,cannot be deleted!");
    }
    else{
      u32 temp_u32 = translate(fs,temp);
      int fcb_id = (int)temp_u32;
      if(fs->volume[4096+fcb_id*32+31]==0xff){
        printf("This is floder, cannotr be delected as file!\n");
      }
      else{
        u32 temp_namesize = check_name_size(s);
        int start_block = (fs->volume[4096+32*fcb_id+28]<<8)+fs->volume[4096+32*fcb_id+29];
        int occupied_blocks = occupied_block_num(fs,fcb_id);
        for(int i=0;i<occupied_blocks;i++){
          clean_superblock(fs,start_block+i);
          clean_file(fs,start_block+i);
        }
        update_current_folder(fs,temp_namesize,fcb_id,2);
        clean_fcb_entry(fs,fcb_id);
        gtime += 1;
      }
      //file_number = file_number-1;
    }
  }
  else if(op==MKDIR){
    int judge = search_file_in_current_folder(fs,s);
    if(judge!=-1){
      printf("Invalid operation, the folder has already existed!\n");
    }
    else{
      int fcbid = (find_next_FCB(fs)-4096)/32;
      //printf("fcbid%d\n",fcbid);
      u32 cur_fol = current_folder;
      u32 namesize = check_name_size(s);
      //printf("namesize:%d\n",namesize);
      //judege if size is enough?
      u32 the_bit = remain_bit(fs);
      u32 remainsize = 32*the_bit;
      if(remainsize<128){
        compact(fs,0);
      }
      the_bit = remain_bit(fs);
      remainsize = 32*the_bit;
      if(remainsize<128){
        printf("No enough space in the volume!\n");
      }
      else{
        fs->volume[4096+32*fcbid+31] = 0xff;
        fs->volume[4096+32*fcbid+30] = 0xff;
        u32 load_block_start = (1<<15)-the_bit;
        //printf("load_block_start:%d\n",load_block_start);
        fs->volume[4096+32*fcbid+28] = load_block_start>>8;
        fs->volume[4096+32*fcbid+29] = load_block_start;
        fs->volume[4096+fcbid*32+24] = 0>>24;
        fs->volume[4096+fcbid*32+25] = 0>>16;
        fs->volume[4096+fcbid*32+26] = 0>>8;
        fs->volume[4096+fcbid*32+27] = 0;
        fs->volume[4096+fcbid*32+22] = gtime>>8;
        fs->volume[4096+fcbid*32+23] = gtime;
        fs->volume[4096+fcbid*32+20] = gtime>>8;
        fs->volume[4096+fcbid*32+21] = gtime;

        int j=0;
        while(s[j]!='\0'){
          fs->volume[4096+fcbid*32+j]=s[j];
          j++;
        }
        fs->volume[4096+fcbid*32+j]='\0';
        j=j+1;
        while(j<=19){
          fs->volume[4096+fcbid*32+j]=0;
          j++;
        }

        fs->volume[36864+32*load_block_start] = 0;
        //printf("cur_fol%d\n",cur_fol);
        fs->volume[36864+32*load_block_start+1] = cur_fol>>8;
        fs->volume[36864+32*load_block_start+2] = cur_fol;
        for(int i=0;i<4;i++){
          set_superblock(fs,load_block_start+i);
        }
        
        update_current_folder(fs,namesize,fcbid,1);
        
        gtime = gtime +1;
      }
    }
  }
  else if(op==CD){
    int judge = search_file_in_current_folder(fs,s);
    if(judge==-1){
      printf("No such directory, cannot CD\n");
    }
    else{
      u32 current_folder_file_start = (fs->volume[4096+32*current_folder+28]<<8)+fs->volume[4096+32*current_folder+29];
      u32 target_fcb = (fs->volume[36864+32*current_folder_file_start+1+2*judge]<<8)+fs->volume[36864+32*current_folder_file_start+2+2*judge];
      current_folder = target_fcb;
    }
  }
  else if(op==RM_RF){
    int ith = search_file_in_current_folder(fs,s);
    u32 fcbid = translate(fs,ith);
    u32 namesize = check_name_size(s);
    delete_all_infloder(fs,fcbid);
    update_current_folder(fs,namesize,fcbid,2);
    delete_single(fs,fcbid);
    gtime = gtime + 1;
  }
}

__device__ void delete_single(FileSystem *fs, u32 fcbid){
  if(fs->volume[4096+32*fcbid+31]==0x0f){
    int start_block = (fs->volume[4096+32*fcbid+28]<<8)+fs->volume[4096+32*fcbid+29];
    int occupied_blocks = occupied_block_num(fs,fcbid);
    for(int i=0;i<occupied_blocks;i++){
      clean_superblock(fs,start_block+i);
      clean_file(fs,start_block+i);
    }
    clean_fcb_entry(fs,fcbid);
  }
  else if(fs->volume[4096+32*fcbid+31]==0xff){
    int start_block = (fs->volume[4096+32*fcbid+28]<<8)+fs->volume[4096+32*fcbid+29];
    for(int i=0;i<4;i++){
      clean_superblock(fs,start_block+i);
      clean_file(fs,start_block+i);
    }
    clean_fcb_entry(fs,fcbid);
  }
}

__device__ void delete_all_infloder(FileSystem *fs,u32 folder_fcbid){
  u32 start_block = (fs->volume[4096+32*folder_fcbid+28]<<8)+fs->volume[4096+32*folder_fcbid+29];
  u32 base_place = 36864+32*start_block;
  for(int i=0;i<50;i++){
    u32 fcbid = (fs->volume[base_place+3+2*i]<<8)+fs->volume[base_place+4+2*i];
    if(fcbid!=0){
      if(fs->volume[4096+32*fcbid+31]==0x0f){
        delete_single(fs,fcbid);
      }
      else if(fs->volume[4096+32*fcbid+31]==0xff){
        delete_all_infloder(fs,fcbid);
        delete_single(fs,fcbid);
      }
    }
  }
}


__device__ u32 parent(FileSystem *fs,u32 fcbid){
  if( (fs->volume[4096+32*fcbid+31]) == 0xff){
    u32 start_block = (fs->volume[4096+32*fcbid+28]<<8) + fs->volume[4096+32*fcbid+29];
    //printf("parent function start block:%d\n",start_block);
    u32 parent_fcbid = (fs->volume[36864+32*start_block+1]<<8) + fs->volume[36864+32*start_block+2];
    return parent_fcbid;
  }
  else{
    printf("This is a file, have no parent!\n");
  }
}


__device__ void print_path(FileSystem *fs,u32 fcbid){
  if (fcbid==0){
    printf("/\n");
  }
  else{
    u32 temp = parent(fs,fcbid);
    //printf("%d\n",temp);
    if(temp==0){
      printf("/");
      uchar* name = &fs->volume[4096+32*fcbid];
      print_string(name);
      printf("\n");
    }
    else{
      uchar* name1 = &fs->volume[4096+32*fcbid];
      uchar* name2 = &fs->volume[4096+32*temp];
      printf("/");
      print_string(name2);
      printf("/");
      print_string(name1);
      printf("\n");
    }
  }
  
}

/*
__device__ void print_path(FileSystem *fs,u32 fcbid){
  u32 temp = parent(fs,fcbid);
  printf("111\n");
  if(fcbid!=0){
    print_path(fs,temp);
    uchar* name = &fs->volume[4096+32*fcbid];
    print_string(name);
  }
  else{
    printf("/");
  }
}
*/


