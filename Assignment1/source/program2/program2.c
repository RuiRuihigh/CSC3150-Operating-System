#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

struct waitid_info {
	pid_t pid;
	uid_t uid;
	int status;
	int cause;
};

struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;
	wait_queue_entry_t child_wait;
	int notask_error;
};

extern pid_t kernel_clone(struct kernel_clone_args *args);
extern long do_wait(struct wait_opts *wo);
extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);
// extern struct filename *getname(const char __user * filename);
extern struct filename *getname_kernel(const char *filename);

static struct task_struct *task;
int status;

int is_normal(int aim)
{
	if ((aim & 0x7f) == 0) {
		return 1;
	} else {
		return 0;
	}
	//return (aim & 0x7f) == 0;
}
int is_stop(int aim)
{
	if ((aim & 0xff) == 0x7f) {
		return 1;
	} else {
		return 0;
	}
	//return (aim & 0xff) == 0x7f;
}
int is_failed(int aim) //signed char
{
	if (((signed char)(((aim & 0x7f) + 1) >> 1)) > 0) {
		return 1;
	} else {
		return 0;
	}
	//return ((signed char)(((aim & 0x7f) + 1) >> 1)) > 0;
}

int new_created_exec(void)
{
	int receive;
	// const char aim_position[] = "/tmp/test"; "/home/vagrant/csc3150/Assignment_1_120090723/source/program2/test"
	const char aim_position[] = "/tmp/test";
	// const char *const argv[] = {aim_position,NULL,NULL};
	// const char *const envp[] = {"HOME=/root",
	// "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",NULL};//????
	struct filename *Myfilename = getname_kernel(aim_position);
	// printk("[program2] : child process");

	// receive = do_execve(Myfilename,argv,envp);
	receive = do_execve(Myfilename, NULL, NULL);
	// printk("the reslut of do_execve: %d \n",receive);
	// printk("receive:\n");
	// printk(receive);
	if (!receive) {
		return 0;
	}
	do_exit(receive);
}

void new_created_wait(pid_t pid)
{
	int L;
	struct wait_opts wo1;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo1.wo_type = type;
	wo1.wo_flags = WEXITED | WSTOPPED;
	wo1.wo_pid = wo_pid;
	wo1.wo_info = NULL;
	wo1.wo_stat = status; //(int __user*)status;
	wo1.wo_rusage = NULL;

	do_wait(&wo1);
	L = wo1.wo_stat;
	if (is_normal(L) == 1) {
		printk("[program2] : get normal signal");
		printk("[program2] : child process normally terminated");
		printk("[program2] : The return signal is %d", L);
	} else if (is_stop(L) == 1) {
		int stopsign = (L & 0xff00) >> 8;
		if (stopsign == 19) {
			printk("[program2] : get SIGSTOP signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 19);
		} else {
			printk("[program2] : some other signals!");
		}
	} else if (is_failed(L) == 1) {
		int tersignal = L & 0x7f;
		if (tersignal == 9) {
			printk("[program2] : get SIGKILL signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 9);
		} else if (tersignal == 1) {
			printk("[program2] : get SIGHUP signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 1);
		} else if (tersignal == 15) {
			printk("[program2] : get SIGTERM signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 15);
		} else if (tersignal == 4) {
			printk("[program2] : get SIGILL signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 4);
		} else if (tersignal == 14) {
			printk("[program2] : get SIGALRM signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 14);
		} else if (tersignal == 7) {
			printk("[program2] : get SIGBUS signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 7);
		} else if (tersignal == 13) {
			printk("[program2] : get SIGPIPE signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 13);
		} else if (tersignal == 5) {
			printk("[program2] : get SIGTRAP signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 5);
		} else if (tersignal == 2) {
			printk("[program2] : get SIGINT signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 2);
		} else if (tersignal == 6) {
			printk("[program2] : get SIGABRT signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 6);
		} else if (tersignal == 3) {
			printk("[program2] : get SIGQUIT signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 3);
		} else if (tersignal == 8) {
			printk("[program2] : get SIGFPE signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 8);
		} else if (tersignal == 11) {
			printk("[program2] : get SIGSEGV signal");
			printk("[program2] : child process terminated");
			printk("[program2] : The return signal is %d", 11);
		} else {
			printk("[program2] : some other signals!");
		}
	} else {
		printk("[program2] : CHILD PROCESS CONTINUED\n");
	}

	put_pid(wo_pid);
	return;
}

// implement fork function
int my_fork(void *argc)
{
	pid_t pid;

	struct kernel_clone_args kca = {
		//.flags = SIGCHLD,
		.exit_signal = SIGCHLD,
		.child_tid = NULL,
		.parent_tid = NULL,
		.stack = (unsigned long)new_created_exec,
		.stack_size = 0,
		.tls = 0,
	};

	/*
  struct kernel_clone_args kca;
  kca.exit_signal = SIGCHLD;
  kca.stack = (unsigned long)new_created_exec;
  kca.stack_size = 0;
  kca.parent_tid = NULL;
  kca.child_tid = NULL;
  kca.tls = 0;
  */
	// set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using kernel_clone or kernel_thread */

	/* execute a test program in child process */

	/* wait until child process terminates */

	pid = kernel_clone(&kca);
	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       (int)current->pid);
	printk("[program2] : child process");
	new_created_wait(pid);
	return 0;
}

static int __init program2_init(void)
{
	printk("[program2] : Module_init {Liang MingRui} {120090723}\n");

	/* write your code here */

	/* create a kernel thread to run my_fork */
	printk("[program2] : module_init create kthread start\n");
	task = kthread_create(&my_fork, NULL, "MyThread");
	if (!IS_ERR(task)) {
		printk("[program2] : Module_init kthread start\n");
		wake_up_process(task);
	}
	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : Module_exit./my\n");
}

module_init(program2_init);
module_exit(program2_exit);
