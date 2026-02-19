__kernel void fast_bitonic_sort_kernel(__global int2* g_data, __local int* l_data) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    int2 data = g_data[global_id];
    l_data[local_id * 2]     = data.x;
    l_data[local_id * 2 + 1] = data.y;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint total_elements = get_local_size(0) * 2;
    uint global_offset = get_group_id(0) * total_elements;

    for (uint stage = 2; stage <= total_elements; stage *= 2) {
        for (uint step = stage / 2; step > 0; step /= 2) {

            uint offset = 31 - clz(step);
            uint left_index = ((local_id >> offset) << (offset + 1)) | (local_id & (step - 1));
            uint right_index = left_index + step;

            int left = l_data[left_index];
            int right = l_data[right_index];

            uint current_global_index = global_offset + left_index;
            bool isAscending = (current_global_index & stage) == 0;

            int min_val = min(left, right);
            int max_val = max(left, right);

            l_data[left_index]  = isAscending ? min_val : max_val;
            l_data[right_index] = isAscending ? max_val : min_val;

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    data.x = l_data[local_id * 2];
    data.y = l_data[local_id * 2 + 1];
    g_data[global_id] = data;
}