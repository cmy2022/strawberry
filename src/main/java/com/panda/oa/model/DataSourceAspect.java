package com.panda.oa.model;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class DataSourceAspect {
	@Around("@annotation(dataSource)")
    public Object switchDataSource(ProceedingJoinPoint joinPoint, DataSource dataSource) throws Throwable {
        try {
            DataSourceKey key = dataSource.value();
            DataSourceContextHolder.setDataSource(key);
            return joinPoint.proceed();
        } finally {
            DataSourceContextHolder.clear();
        }
    }
}