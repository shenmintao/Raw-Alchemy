"""
测光策略模块
使用策略模式实现不同的测光算法
"""
import numpy as np
from typing import Protocol, Optional

try:
    from .logger import Logger
    from . import utils
except ImportError:
    from raw_alchemy.logger import Logger
    from raw_alchemy import utils

class MeteringStrategy(Protocol):
    """测光策略接口"""
    
    def calculate_gain(
        self,
        img_linear: np.ndarray,
        source_colorspace,
        target_gray: float = 0.18,
        logger: Optional[Logger] = None
    ) -> float:
        """
        计算曝光增益
        
        Args:
            img_linear: 线性图像数据
            source_colorspace: 源色彩空间
            target_gray: 目标灰度值
            logger: 日志处理器
        
        Returns:
            float: 曝光增益值
        """
        ...


class AverageMeteringStrategy:
    """平均测光策略（几何平均）"""
    
    def calculate_gain(
        self,
        img_linear: np.ndarray,
        source_colorspace,
        target_gray: float = 0.18,
        logger: Optional[Logger] = None
    ) -> float:

        sample = utils.get_subsampled_view(img_linear)
        coeffs = utils.get_luminance_coeffs(source_colorspace)
        luminance = np.dot(sample, coeffs)
        
        # 确保亮度值为正，避免对数运算中的无效值
        luminance = np.maximum(luminance, 1e-10)
        avg_log_lum = np.mean(np.log(luminance + 1e-6))
        avg_lum = np.exp(avg_log_lum)
        
        if avg_lum < 0.0001:
            gain = 1.0
        else:
            gain = target_gray / avg_lum
        
        gain = np.clip(gain, 1.0, 50.0)
        
        if logger:
            logger.info(f"  ⚖️  [Auto Exposure] Avg Gain: {gain:.4f}")
        
        return gain


class CenterWeightedMeteringStrategy:
    """中央重点测光策略"""
    
    def calculate_gain(
        self,
        img_linear: np.ndarray,
        source_colorspace,
        target_gray: float = 0.18,
        logger: Optional[Logger] = None
    ) -> float:

        sample = utils.get_subsampled_view(img_linear)
        coeffs = utils.get_luminance_coeffs(source_colorspace)
        luminance = np.dot(sample, coeffs)
        
        h, w = luminance.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        sigma = min(h, w) / 2
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        weights = np.exp(-dist_sq / (2 * sigma**2))
        
        weighted_avg_lum = np.average(luminance, weights=weights)
        
        if weighted_avg_lum < 1e-6:
            gain = 1.0
        else:
            gain = target_gray / weighted_avg_lum
        
        gain = np.clip(gain, 0.1, 100.0)
        
        if logger:
            logger.info(f"  ⚖️  [Auto Exposure] Center-Weighted Gain: {gain:.4f}")
        
        return gain


class HighlightSafeMeteringStrategy:
    """高光保护测光策略（ETTR）"""
    
    def calculate_gain(
        self,
        img_linear: np.ndarray,
        source_colorspace,
        target_gray: float = 0.18,
        logger: Optional[Logger] = None
    ) -> float:

        sample = utils.get_subsampled_view(img_linear)
        max_vals = np.max(sample, axis=2)
        high_percentile = np.percentile(max_vals, 99.0)
        
        target_high = 0.9
        if high_percentile < 1e-6:
            gain = 1.0
        else:
            gain = target_high / high_percentile
        
        if logger:
            logger.info(f"  🛡️  [Auto Exposure] Highlight Safe Gain: {gain:.4f}")
        
        return gain


class HybridMeteringStrategy:
    """混合测光策略（平均 + 高光限制）"""
    
    def calculate_gain(
        self,
        img_linear: np.ndarray,
        source_colorspace,
        target_gray: float = 0.18,
        logger: Optional[Logger] = None
    ) -> float:
        
        sample = utils.get_subsampled_view(img_linear)
        coeffs = utils.get_luminance_coeffs(source_colorspace)
        luminance = np.dot(sample, coeffs)
        
        # 确保亮度值为正，避免对数运算中的无效值
        luminance = np.maximum(luminance, 1e-10)
        avg_log_lum = np.mean(np.log(luminance + 1e-6))
        avg_lum = np.exp(avg_log_lum)
        base_gain = target_gray / (avg_lum + 1e-6)
        
        max_vals = np.max(sample, axis=2)
        p99 = np.percentile(max_vals, 99.0)
        
        potential_peak = p99 * base_gain
        max_allowed_peak = 6.0
        
        if potential_peak > max_allowed_peak:
            limited_gain = max_allowed_peak / p99
            if logger:
                logger.info(f"  🛡️  [Auto Exposure] Hybrid limited. (Desired: {base_gain:.2f} -> Actual: {limited_gain:.2f})")
            gain = limited_gain
        else:
            gain = base_gain
        
        gain = np.clip(gain, 0.1, 100.0)
        
        if logger:
            logger.info(f"  ⚖️  [Auto Exposure] Hybrid Gain: {gain:.4f}")
        
        return gain


class MatrixMeteringStrategy:
    """矩阵/评价测光策略"""
    
    def calculate_gain(
        self,
        img_linear: np.ndarray,
        source_colorspace,
        target_gray: float = 0.18,
        logger: Optional[Logger] = None
    ) -> float:
        
        sample = utils.get_subsampled_view(img_linear)
        h, w, _ = sample.shape
        
        coeffs = utils.get_luminance_coeffs(source_colorspace)
        luminance = np.dot(sample, coeffs)
        
        grid_size = 7
        grid_h, grid_w = h // grid_size, w // grid_size
        
        grid_lums = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                cell = luminance[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                if cell.size > 0:
                    grid_lums[i, j] = np.mean(cell)
        
        weights = np.ones((grid_size, grid_size))
        
        # 中心偏置
        y, x = np.ogrid[:grid_size, :grid_size]
        center_y, center_x = (grid_size - 1) / 2.0, (grid_size - 1) / 2.0
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        sigma = grid_size / 2.5
        center_bias = np.exp(-dist_sq / (2 * sigma**2))
        weights *= (1 + center_bias * 1.5)
        
        # 高光抑制
        lum_percentile_90 = np.percentile(grid_lums, 90)
        highlight_zones = grid_lums > lum_percentile_90
        weights[highlight_zones] *= 0.2
        
        # 暗部关注
        lum_percentile_10 = np.percentile(grid_lums, 10)
        shadow_zones = grid_lums < lum_percentile_10
        weights[shadow_zones] *= 1.2
        
        weighted_avg_lum = np.average(grid_lums, weights=weights)
        
        if weighted_avg_lum < 1e-6:
            gain = 1.0
        else:
            gain = target_gray / weighted_avg_lum
        
        # 保护性削减
        max_vals = np.max(sample, axis=2)
        p99 = np.percentile(max_vals, 99.0)
        potential_peak = p99 * gain
        max_allowed_peak = 6.0
        
        if potential_peak > max_allowed_peak:
            limited_gain = max_allowed_peak / p99
            if logger:
                logger.info(f"  🛡️  [Auto Exposure] Matrix limited. (Desired: {gain:.2f} -> Actual: {limited_gain:.2f})")
            gain = limited_gain
        
        gain = np.clip(gain, 0.1, 100.0)
        
        if logger:
            logger.info(f"  🤖 [Auto Exposure] Matrix Gain: {gain:.4f}")
        
        return gain


# 策略注册表
METERING_STRATEGIES = {
    'average': AverageMeteringStrategy(),
    'center-weighted': CenterWeightedMeteringStrategy(),
    'highlight-safe': HighlightSafeMeteringStrategy(),
    'hybrid': HybridMeteringStrategy(),
    'matrix': MatrixMeteringStrategy(),
}


def get_metering_strategy(mode: str) -> MeteringStrategy:
    """
    获取测光策略
    
    Args:
        mode: 测光模式名称
    
    Returns:
        MeteringStrategy: 对应的测光策略实例
    
    Raises:
        ValueError: 如果模式不存在
    """
    strategy = METERING_STRATEGIES.get(mode)
    if strategy is None:
        raise ValueError(f"Unknown metering mode: {mode}")
    return strategy


def apply_auto_exposure(
    img_linear: np.ndarray,
    source_colorspace,
    metering_mode: str = 'hybrid',
    target_gray: float = 0.18,
    logger: Optional[Logger] = None
) -> np.ndarray:
    """
    应用自动曝光
    
    Args:
        img_linear: 线性图像数据
        source_colorspace: 源色彩空间
        metering_mode: 测光模式
        target_gray: 目标灰度值
        logger: 日志处理器
    
    Returns:
        np.ndarray: 调整后的图像
    """

    strategy = get_metering_strategy(metering_mode)
    gain = strategy.calculate_gain(img_linear, source_colorspace, target_gray, logger)
    utils.apply_gain_inplace(img_linear, float(gain))
    
    return img_linear
