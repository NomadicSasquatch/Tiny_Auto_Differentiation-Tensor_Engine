#include <stdio.h>

int main() {
    #if defined(__STDC_VERSION__)
        if (__STDC_VERSION__ == 201710L) printf("C17/C18\n");
        else if (__STDC_VERSION__ == 201112L) printf("C11\n");
        else if (__STDC_VERSION__ == 199901L) printf("C99\n");
        else if (__STDC_VERSION__ == 199409L) printf("C95\n");
        else printf("C Standard Version: %ld (Unknown)\n", __STDC_VERSION__);
    #elif defined(__STDC__)
        printf("C89/C90 (ANSI C)\n");
    #else
        printf("Pre-standard C (K&R C)\n");
    #endif
    return 0;
}