
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

my_queue_t que;
my_queue_t *my_que;
pthread_mutex_t mutex_exe;
pthread_cond_t cond_exe;
pthread_cond_t cond_exe_extra;
void *async_process(void *idp);
void finish_rest(void);
void wraper_finish(int sub);
void *wake(void *ar);

void async_init(int num_threads) {
    my_que = &que;
    pthread_t threads[num_threads+1];
    pthread_mutex_init(&mutex_exe,NULL);
    pthread_cond_init(&cond_exe,NULL);
    pthread_cond_init(&cond_exe_extra,NULL);
    for(int i=0;i<num_threads;i++){
        pthread_create(&threads[i],NULL,async_process,NULL);
    }
    pthread_create(&threads[num_threads],NULL,wake,NULL);
    return;
    /** TODO: create num_threads threads and initialize the thread pool **/
}

void async_run(void (*hanlder)(int), int args) {
    my_item_t *an_item = (my_item_t*)malloc(sizeof(my_item_t));
    an_item->sth_number = args;
    an_item->func = hanlder;
    DL_APPEND(my_que->head,an_item);
    my_que->size ++;
    pthread_cond_broadcast(&cond_exe);
    //hanlder(args);
    /** TODO: rewrite it to support thread pool **/
}

void *wake(void *ar){
    while(1){
        pthread_mutex_lock(&mutex_exe);
        pthread_cond_wait(&cond_exe_extra,&mutex_exe);
        pthread_cond_broadcast(&cond_exe);
        pthread_mutex_unlock(&mutex_exe);
    }
}

void *async_process(void *idp){
    while(1){
        pthread_mutex_lock(&mutex_exe);
        pthread_cond_wait(&cond_exe,&mutex_exe);
        if(my_que->head==NULL){//指针到底指向啥
            pthread_mutex_unlock(&mutex_exe);
        }
        else{
            my_item_t *one_item;
            void (*func1)(int);
            one_item = my_que->head;
            DL_DELETE(my_que->head,my_que->head);
            my_que->size --;
            int num_th = one_item->sth_number;
            func1 = one_item->func;
            free(one_item);
            func1(num_th);   
            pthread_mutex_unlock(&mutex_exe);        
            
            int re;
            re = pthread_mutex_trylock(&mutex_exe);
            if(re==0){
                if(my_que->size!=0){
                    pthread_cond_signal(&cond_exe_extra);
                }
                pthread_mutex_unlock(&mutex_exe);
            }
        }
    }
}
//    for(int i=0;i<sub;i++){
/*
void finish_rest(void){
    while(my_que->size!=0){
        //pthread_mutex_lock(&mutex_exe);
        if(my_que->head!=NULL){
        my_item_t *onee_item;
        void (*func2)(int);
        onee_item = my_que->head;
        DL_DELETE(my_que->head,my_que->head);
        my_que->size --;
        int num_th = onee_item->sth_number;
        func2 = onee_item->func;
        free(onee_item);
        func2(num_th); 
        //pthread_mutex_unlock(&mutex_exe); 
        finish_rest();
        }
        else{
            break;
        }
    }
}
*/