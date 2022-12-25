#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>


#define ROW 10
#define COLUMN 50 



struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 

pthread_mutex_t mutex1;
int status=0b00;//0b00:default 0b01:loss game 0b10:quit 0b11:game win
char map[ROW+10][COLUMN] ; 



// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void *logs_move( void *t ){
	/*  Move the logs  */
	/*  Check keyboard hits, to change frog's position or quit the game. */
	/*  Check game's status  */
	/*  Print the map on the screen  */
	while (status==0b00){
		usleep(50000);
		//sleep(1);
		pthread_mutex_lock(&mutex1);
		for(int i=1;i<ROW;i++){
			//left
			if(i%2==1){
				for(int k=0;k<COLUMN-1;k++){
					if(map[i][k]=='='){
						map[i][(k+COLUMN-2)%(COLUMN-1)]='=';
						map[i][k]=' ';
					}
					if(map[i][k]=='0'){
						map[i][(k+COLUMN-2)%(COLUMN-1)]='0';
						map[i][k]=' ';
						frog.y = frog.y-1;
						if(frog.y<0 || frog.y>COLUMN-2){
							status=0b01;
						}
					}
				}
			}
			
			//right
			else{
				for(int l=COLUMN-2;l>=0;l--){
					if(map[i][l]=='='){
						map[i][(l+1)%(COLUMN-1)]='=';
						map[i][l]=' ';
					}
					if(map[i][l]=='0'){
						map[i][(l+1)%(COLUMN-1)]='0';
						map[i][l]=' ';
						frog.y = frog.y+1;
						if(frog.y<0 || frog.y>COLUMN-2){
							status=0b01;
						}
					}
				}
			}
		}
		puts("\033[H\033[2J");
        for(int i = 0; i <= ROW; ++i){
            puts(map[i]);
        }
		pthread_mutex_unlock(&mutex1);
		
	}
	pthread_exit(NULL);
	
}

void *frog_move(void *f){
	while(status==0b00){
		usleep(50000);
		pthread_mutex_lock(&mutex1);
		if(kbhit()==1){
			int key = getchar();
			if(key==87 || key ==119){//W/w
				if(frog.x-1==0){
					status=0b11;
				}
				else if(frog.x==ROW){
					//system("pause");
					map[frog.x][frog.y]='|';
					frog.x = frog.x-1;
					if(map[frog.x][frog.y]=='='){
						map[frog.x][frog.y]='0';
					}
					else{
						status=0b01;
					}
				}
				else{
					map[frog.x][frog.y]='=';
					frog.x = frog.x-1;
					if(map[frog.x][frog.y]=='='){
						map[frog.x][frog.y]='0';
					}
					else{
						status=0b01;
					}
				}
			}
			else if(key==83 || key==115){//S/s
				
				if(frog.x==ROW-1){
					map[frog.x][frog.y]='=';
					frog.x = frog.x+1;
					map[frog.x][frog.y]='0';
				}
				else if(frog.x!=ROW){
					map[frog.x][frog.y]='=';
					frog.x = frog.x+1;
					if(map[frog.x][frog.y]=='='){
						map[frog.x][frog.y]='0';
					}
					else{
						status=0b01;
					}
				}
				else{
					//system("pause");
				}
			}
			else if(key==65 || key==97){//A/a
				if(frog.x==ROW){
					if(frog.y==0);
					else{
						//system("pause");
						map[frog.x][frog.y]='|';
						frog.y = frog.y-1;
						map[frog.x][frog.y]='0';
					}
				}
				else{
					if(frog.y==0){
						status=0b01;
					}
					else{
						map[frog.x][frog.y]='=';
						frog.y = frog.y-1;
						if(map[frog.x][frog.y]=='='){
							map[frog.x][frog.y]='0';
						}
						else{
							status=0b01;
						}
					}
				}
			}
			else if(key==68 || key==100){//D/d
				if(frog.x==ROW){
					if(frog.y==COLUMN-2);
					else{
						//system("pause");
						map[frog.x][frog.y]='|';
						frog.y = frog.y+1;
						map[frog.x][frog.y]='0';
					}
				}
				else{
					if(frog.y==COLUMN-1){
						status=0b01;
					}
					else{
						map[frog.x][frog.y]='=';
						frog.y = frog.y+1;
						if(map[frog.x][frog.y]=='='){
							map[frog.x][frog.y]='0';
						}
						else{
							status=0b01;
						}
					}
				}
			}
			else if(key==81 || key==113){
				status=0b10;
			}
			else;
			//puts("\033[H\033[2J");
        	//for(int i = 0; i <= ROW; ++i){
            //	puts(map[i]);
        	//}
			//usleep(50000);
		}
		pthread_mutex_unlock(&mutex1);
	}
	pthread_exit(NULL);
}

int main( int argc, char *argv[] ){
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 
	//decide the decision of the original logs
	int random_start;
	int loglen = 15;
	for(int i=1;i<ROW;i++){
		random_start = rand()%COLUMN;
		for (int k=0;k<loglen;k++){
			map[i][(random_start+k)%(COLUMN-1)]='=';
		}
	}
	//Print the map into screen
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );

	/*  Create pthreads for wood move and frog control.  */
	/*  Display the output for user: win, lose or quit.  */
	pthread_t logthread,frogthread;
	pthread_mutex_init(&mutex1,NULL);
	pthread_create(&logthread,NULL,logs_move,NULL);
	pthread_create(&frogthread,NULL,frog_move,NULL);
	pthread_join(logthread,NULL);
	pthread_join(frogthread,NULL);
	
	printf("\033[H\033[2J");
	if(status==0b01){
		printf("You lose the game!!\n");
	}
	else if(status==0b10){
		printf("You exit the game.\n");
	}
	else if(status==0b11){
		printf("You win the game!!\n");
	}
	else{
		printf("You exit in the default status!!!\n");
	}


	pthread_mutex_destroy(&mutex1);
	pthread_exit(NULL);
	return 0;

}
