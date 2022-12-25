#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[])
{
	/* fork a child process */

	/* execute test program */

	/* wait for child process terminates */

	/* check child process'  termination status */

	pid_t pid;
	int status;
	printf("Process start to fork\n");
	pid = fork();
	if (pid == -1) {
		perror("fork");
		exit(1);
	} else {
		if (pid == 0) {
			printf("I'm the Child Process, my pid = %d\n",
			       getpid());
			int i;
			char *arg[argc];
			for (i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;
			printf("Child process start to execute test program:\n");
			//printf("------------CHILD PROCESS START------------\n");
			execve(arg[0], arg, NULL);

			printf("Continue to run original child process!\n");
			perror("execve");
			exit(EXIT_FAILURE);

		} else {
			printf("I'm the Parent Process, my pid = %d\n",
			       getpid());

			waitpid(-1, &status, WUNTRACED);
			printf("Parent process receives SIGCHLD signal\n");
			if (WIFEXITED(status)) {
				printf("Normal termination with EXIT STATUS = %d\n",
				       0);
			} else if (WIFSIGNALED(status)) {
				int signal = WTERMSIG(status);
				if (signal == 9) {
					//printf("child process was terminated by kill signal\n");
					printf("child process get SIGKILL signal\n");
				} else if (signal == 1) {
					//printf("child process was terminated by hung up signal\n");
					printf("child process get SIGHUP signal\n");
				} else if (signal == 15) {
					//printf("child process was terminated by terminate signal\n");
					printf("child process get SIGTERM signal\n");
				} else if (signal == 4) {
					//printf("child process was terminated by illegal instruction signal\n");
					printf("child process get SIGILL signal\n");
				} else if (signal == 14) {
					//printf("child process was terminated by alarm signal\n");
					printf("child process get SIGALRM signal\n");
				} else if (signal == 7) {
					//printf("child process was terminated by bus error signal\n");
					printf("child process get SIGBUS signal\n");
				} else if (signal == 13) {
					//printf("child process was terminated by broken pipe signal\n");
					printf("child process get SIGPIPE signal\n");
				} else if (signal == 5) {
					//printf("child process was terminated by trap signal\n");
					printf("child process get SIGTRAP signal\n");
				} else if (signal == 2) {
					//printf("child process was terminated by interrupt signal\n");
					printf("child process get SIGINT signal\n");
				} else if (signal == 6) {
					//printf("child process was terminated by abort signal\n");
					printf("child process get SIGABRT signal\n");
				} else if (signal == 3) {
					//printf("child process was terminated by quit signal\n");
					printf("child process get SIGQUIT signal\n");
				} else if (signal == 8) {
					//printf("child process was terminated by floating signal\n");
					printf("child process get SIGFPE signal\n");
				} else if (signal == 11) {
					//printf("child process was terminated by segment_fault signal\n");
					printf("child process get SIGSEGV signal\n");
				} else {
				}
			} else if (WIFSTOPPED(status)) {
				//printf("child process was terminated by stop signal\n");
				printf("child process get SIGSTOP signal\n");
			} else {
				printf("CHILD PROCESS CONTINUED\n");
			}
			exit(0);
		}
	}
}
