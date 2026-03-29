import { useQuery } from '@tanstack/react-query';
import api from '@/src/lib/api-client';

export const useDashboardData = () => {
    return useQuery({
        queryKey: ['dashboard-stats'],
        queryFn: async () => {
            const { data } = await api.get('/dashboard/stats');
            return data;
        },
        staleTime: 1000 * 60 * 5, // 5 minutes
    });
};