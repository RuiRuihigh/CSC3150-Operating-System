#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>

typedef struct my_item {
  /* TODO: More stuff here, maybe? */
  struct my_item *next;
  struct my_item *prev;
  int sth_number;
  void (*func)(int);
} my_item_t;

typedef struct my_queue {
  int size;
  my_item_t *head;
  /* TODO: More stuff here, maybe? */
} my_queue_t;

void async_init(int);
void async_run(void (*fx)(int), int args);

extern my_queue_t que;
extern my_queue_t *my_que;
//extern my_item_t *item_pt;
extern pthread_mutex_t mutex_exe;
//extern pthread_mutex_t mutex_exe_extra;
extern pthread_cond_t cond_exe;
extern pthread_cond_t cond_exe_extra;
#endif
