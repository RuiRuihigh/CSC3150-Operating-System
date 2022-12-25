#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

#define PWD 3
#define MKDIR 4
#define CD 5
#define CD_P 6
#define RM_RF 7

struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
};


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);


//my function declear
__device__ void set_root(FileSystem *fs);
__device__ void volume_init(FileSystem *fs);
__device__ int occupied_block_num(FileSystem *fs,int fcb_id);
__device__ int find_next_FCB(FileSystem *fs);
__device__ int search_file(FileSystem *fs, char *s);
__device__ void clean_superblock(FileSystem *fs,int blockadd);
__device__ void clean_file(FileSystem *fs,int blockadd);
__device__ void set_superblock(FileSystem *fs,int blockadd);
__device__ u32 remain_bit(FileSystem *fs);
__device__ u32 find_fcb(FileSystem *fs,u32 block_id);
__device__ u32 find_first_1(FileSystem *fs,u32 startplace);
__device__ u32 find_first_0(FileSystem *fs,u32 startplace);
__device__ void move(FileSystem *fs,u32 from_block,u32 to_block);
__device__ void compact(FileSystem *fs,u32 start_place);
__device__ void print_name(FileSystem *fs,u32 fcb_id);
__device__ void print_name_size(FileSystem *fs,u32 fcb_id,int size);
__device__ void print_string(uchar *s);
__device__ void clean_fcb_entry(FileSystem *fs,int fcb_id);
//__device__ void fs_gsys(FileSystem *fs,int op);
__device__ void print_path(FileSystem *fs,u32 fcbid);
__device__ u32 parent(FileSystem *fs,u32 fcbid);
__device__ void init_folder(FileSystem *fs,u32 fcbid);
__device__ u32 check_name_size(char* s);
__device__ void update_current_folder(FileSystem *fs,u32 name_size,u32 child_fcbid,int mode);
__device__ int search_file_in_current_folder(FileSystem *fs,char* s);
__device__ u32 translate(FileSystem *fs,int i);
__device__ void delete_all_infloder(FileSystem *fs,u32 folder_fcbid);
__device__ void delete_single(FileSystem *fs, u32 fcbid);

#endif