function cell_removed = XxRemoveCellItems(cell_raw, items_to_remove)

nitems = max(size(cell_raw));
n_to_remove = max(size(items_to_remove));
flag = ones(size(cell_raw));
for i = 1:nitems
    cur_item = cell_raw{i};
    for j = 1:n_to_remove
        if strcmp(cur_item,items_to_remove{j})
            flag(i) = 0;
            break;
        end
    end
end
cell_removed = cell_raw(logical(flag));