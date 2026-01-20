#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <unistd.h>

atomic_int counter = 0;

void* thread_func(void* arg) {
    (void)arg;
    for (int i = 0; i < 1000000; i++) {
        atomic_fetch_add(&counter, 1);
    }
    return NULL;
}

int main(void) {
    pthread_t t1, t2;

    if (pthread_create(&t1, NULL, thread_func, NULL) != 0) {
        perror("pthread_create t1");
        exit(EXIT_FAILURE);
    }
    if (pthread_create(&t2, NULL, thread_func, NULL) != 0) {
        perror("pthread_create t2");
        exit(EXIT_FAILURE);
    }

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Atomic counter final value: %d\n", counter);

    size_t length = 4096;
    void *addr = mmap(
        NULL,
        length,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1, 0
    );
    if (addr == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    const char *msg = "Hello from mmap in WSL!";
    strncpy((char*)addr, msg, length);

    printf("Mapped memory contains: \"%s\"\n", (char*)addr);
    
    if (munmap(addr, length) != 0) {
        perror("munmap");
        exit(EXIT_FAILURE);
    }

    return 0;
}