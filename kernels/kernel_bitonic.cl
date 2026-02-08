__kernel void bitonic_sort_kernel(__global int* data, int stage, int step) {
    int cur_id = get_global_id(0);
    int cur_partner = cur_id + step;

    if ((cur_id & step) == 0) {
        bool isAscending = (cur_id & stage) == 0;
        int left  = data[cur_id];
        int right = data[cur_partner];
        if (isAscending) {
            if (left > right) {
               data[cur_id] = right;
               data[cur_partner] = left;
            }
        }
        else {
            if (left < right) {
                data[cur_id] = right;
                data[cur_partner] = left;
            }
        }
    }
}