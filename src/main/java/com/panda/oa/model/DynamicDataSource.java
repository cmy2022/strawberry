package com.panda.oa.model;

import org.springframework.jdbc.datasource.lookup.AbstractRoutingDataSource;

/**
 * @panda.chen
 * 动态路由数据源
 */
public class DynamicDataSource extends AbstractRoutingDataSource {

	@Override
	protected Object determineCurrentLookupKey() {
		return DataSourceContextHolder.getDataSource();
	}

}
