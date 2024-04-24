function [sentinel_bman] = bellman_ford_mat(chosen_grid, meshes, start, graph, sentinels)
    INF = inf;
    dist = ones(1, meshes) * INF;
    dist(start) = 0;
    n = length(graph);
    
    % Apply Bellman ford algorithm 
    for i = 1:(n - 1)
        for u = 1:n
            for v = 1:n
                if graph(u, v) ~= 0
                    if dist(u) + graph(u, v) < dist(v)
                        dist(v) = dist(u) + graph(u, v);
                    end
                end
            end
        end
    end

    sentinel_bman = cell(1, length(sentinels));  % Initialize as a cell array
    
    for i = 1:length(sentinels)
        x = sentinels(i, 1);
        y = sentinels(i, 2);
        
        % Calculate the corresponding mesh index
        mesh_index = floor((x - 10) / 20) + floor((y - 10) / 20) * chosen_grid + 1;
        % fprintf('sentinel coordinates (%d, %d) and mesh index is %d\n', x,y,mesh_index);

        if mesh_index <= length(graph)
            if dist(mesh_index) == INF
                sentinel_bman{i} = 999; % Assign to cell element
            else
                sentinel_bman{i} = dist(mesh_index); % Assign to cell element
            end
        else
            disp('Mesh index exceeds the length of the graph.');
        end
    end
end