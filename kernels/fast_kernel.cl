__kernel void fast_bitonic_sort_kernel(__global int* g_data, __local int* l_data, int local_work_group_size) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    l_data[local_id] = g_data[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stage = 2; stage <= local_work_group_size; stage *= 2) {
        for (int step = stage / 2; step > 0; step /= 2) {
            int cur_partner = local_id + step;

            if ((local_id & step) == 0) {
                bool isAscending = (global_id & stage) == 0;
                int left  = l_data[local_id];
                int right = l_data[cur_partner];
                if (isAscending) {
                    if (left > right) {
                        l_data[local_id] = right;
                        l_data[cur_partner] = left;
                    }
                }
                else {
                    if (left < right) {
                        l_data[local_id] = right;
                        l_data[cur_partner] = left;
                    }
                }
            }
        barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    g_data[global_id] = l_data[local_id];
}