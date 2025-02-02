'''
    Python Mapper script
    Generated by the Python Mapper GUI
'''

import mapper
import numpy as np
import matplotlib.pyplot as plt

for a in range(2):
    for q in range(13):
        '''
            Step 1: Input
        '''
        alg = 'SIFT'
        if(a == 1):
            alg = 'SURF'
        filename = 'C:/Users/Shanik/Documents/uni/Computer Science/Year 3/project/python/numpy vectors/sunset_vectors/sunset_' + alg + '_' + str(q+1) + '.npy'
        savepath = '../mapper outputs/sunset/' + alg + '/euc, ecc 2, uni 15 50, big 0 10/sunset_' + alg + '_output_' + str(q+1) +'.png'
        scalesave = '../mapper outputs/sunset/' + alg + '/euc, ecc 2, uni 15 50, big 0 10/sunset_' + alg + '_scale_graph_' + str(q+1) +'.png'
        data = np.load(filename).astype(np.float)
        # Preprocessing
        point_labels = None
        mask = None
        Gauss_density = mapper.filters.Gauss_density
        kNN_distance  = mapper.filters.kNN_distance
        crop = mapper.crop
        # Custom preprocessing code

        # End custom preprocessing code
        data, point_labels = mapper.mask_data(data, mask, point_labels)
        '''
            Step 2: Metric
        '''
        intrinsic_metric = False
        if intrinsic_metric:
            is_vector_data = data.ndim != 1
            if is_vector_data:
                metric = Euclidean
                if metric != 'Euclidean':
                    raise ValueError('Not implemented')
            data = mapper.metric.intrinsic_metric(data, k=1, eps=1.0)
        is_vector_data = data.ndim != 1
        '''
            Step 3: Filter function
        '''
        if is_vector_data:
            metricpar = {'metric': 'euclidean'}
            f = mapper.filters.eccentricity(data,
                metricpar=metricpar,
                exponent=2.0)
        else:
            f = mapper.filters.eccentricity(data,
                exponent=2.0)
        # Filter transformation
        '''
        mask = None
        crop = mapper.crop
        # Custom filter transformation

        # End custom filter transformation
        '''
        '''
            Step 4: Mapper parameters
        '''
        cover = mapper.cover.cube_cover_primitive(intervals=15, overlap=50.0)
        cluster = mapper.average_linkage()
        if not is_vector_data:
            metricpar = {}
        mapper_output = mapper.mapper(data, f,
            cover=cover,
            cluster=cluster,
            point_labels=point_labels,
            cutoff=None,
            metricpar=metricpar)
        cutoff = mapper.cutoff.variable_exp_gap(maxcluster=10, exponent=0.0)
        mapper_output.cutoff(cutoff, f, cover=cover, simple=False)
        mapper_output.draw_scale_graph()
        plt.savefig(scalesave)
        '''
            Step 5: Display parameters
        '''
        # Node coloring
        node_color = None
        point_color = None
        name = "default"
        # End custom node coloring
        #node_color = mapper_output.postprocess_node_color(node_color, point_color, point_labels)
        minsizes = []
        mapper_output.draw_2D(minsizes=minsizes,
            node_color=node_color,
            node_color_scheme=name)

        plt.savefig(savepath)
        # plt.show()
