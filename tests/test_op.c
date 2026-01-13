#include "unity.h"
#include "op.h"

void setUp(void)   {}
void tearDown(void){}

void test_add(void) {
    TEST_ASSERT_EQUAL_INT(5, add(2, 3));
}

int main(void) {
    UnityBegin("op.c");
    RUN_TEST(test_add);
    return UnityEnd();
}
