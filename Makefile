CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -Werror -g
CPPFLAGS := -Iinclude
LDLIBS := -lm

OBJDIR := build/obj
BINDIR := build/bin

CORE_SRCS := \
  src/core/arena.c src/core/graph.c src/core/prob_helper.c \
  src/core/registry.c src/core/tensor.c src/core/utils.c

DATA_SRCS := src/data/dataset.c
NN_SRCS := src/nn/loss.c src/nn/nn.c src/nn/optim.c

OPS_SRCS = \
  src/ops/add.c src/ops/matmul.c src/ops/mul.c \
  src/ops/relu.c src/ops/softmax.c src/ops/sub.c

LIB_SRCS := $(CORE_SRCS) $(DATA_SRCS) $(NN_SRCS) $(OPS_SRCS)
TRAIN_SRC := src/model/train.c

TRAIN_OBJS := $(patsubst %.c,$(OBJDIR)/%.o,$(LIB_SRCS) $(TRAIN_SRC))

.PHONY: all clean run selftest-arena selftest-tensor selftest-registry selftest-add selftest-sub selftest-mul selftest-matmul selftest-relu selftest-softmax

all: $(BINDIR)/train

# generic object build rule (keeps directory structure under build/obj/)
$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BINDIR)/train: $(TRAIN_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(TRAIN_OBJS) -o $@ $(LDLIBS)

run: $(BINDIR)/train
	./$(BINDIR)/train $(ARGS)

clean:
	rm -rf build

# self smoketests compile the SAME .c as a standalone program in the guarded main blokcs
selftest-utils: $(BINDIR)/utils_selftest
	./$(BINDIR)/utils_selftest

$(BINDIR)/utils_selftest: src/core/utils.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DUTILS_SELFTEST_MAIN $< -o $@


selftest-arena: $(BINDIR)/arena_selftest
	./$(BINDIR)/arena_selftest

$(BINDIR)/arena_selftest: src/core/arena.c src/core/utils.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DARENA_SELFTEST_MAIN $^ -o $@


selftest-tensor: $(BINDIR)/tensor_selftest
	./$(BINDIR)/tensor_selftest

$(BINDIR)/tensor_selftest: src/core/tensor.c src/core/arena.c src/core/utils.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DTENSOR_SELFTEST_MAIN $^ -o $@

selftest-graph: $(BINDIR)/graph_selftest
	./$(BINDIR)/graph_selftest

$(BINDIR)/graph_selftest: src/core/tensor.c src/core/arena.c src/core/utils.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DGRAPH_SELFTEST_MAIN $^ -o $@

# better way to aggregate? OPS START
selftest-add: $(BINDIR)/add_selftest
	./$(BINDIR)/add_selftest

$(BINDIR)/add_selftest: src/ops/add.c src/core/tensor.c src/core/arena.c src/core/utils.c src/core/registry.c src/core/graph.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DADD_SELFTEST_MAIN $^ -o $@ $(LDLIBS)

selftest-registry: \
	selftest-add \
# 	selftest-sub \
# 	selftest-mul \
# 	selftest-matmul \
# 	selftest-relu \
# 	selftest-softmax
# OPS END