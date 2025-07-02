package com.panda.oa.model;

/**
 * @panda.chen
 * 数据源上下文
 */
public class DataSourceContextHolder {
	
	private static final ThreadLocal<DataSourceKey> CONTEXT = new ThreadLocal<>();
	
    public static void setDataSource(DataSourceKey type) {
        CONTEXT.set(type);
    }
    
    public static DataSourceKey getDataSource() {
        return CONTEXT.get() == null ? DataSourceKey.FIRST : CONTEXT.get();
    }
    
    public static void clear() {
        CONTEXT.remove();
    }
}
