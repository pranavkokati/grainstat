import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime

from ..main import GrainAnalyzer
from ..export.data_export import DataExporter
from ..export.reports import ReportGenerator


class BatchProcessor:

    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.analyzer = GrainAnalyzer()
        self.exporter = DataExporter()
        self.reporter = ReportGenerator()

    def process_directory(self, input_dir: str, output_dir: str,
                          pattern: str = "*.tif*", scale: float = 1.0,
                          analysis_params: Optional[Dict[str, Any]] = None,
                          save_individual_results: bool = True,
                          generate_summary: bool = True) -> Dict[str, Any]:

        # Find all image files
        image_files = self._find_image_files(input_dir, pattern)

        if not image_files:
            raise ValueError(f"No image files found in {input_dir} with pattern {pattern}")

        print(f"Found {len(image_files)} images to process")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set default analysis parameters
        if analysis_params is None:
            analysis_params = {
                'min_area': 50,
                'gaussian_sigma': 1.0,
                'threshold_method': 'otsu',
                'use_watershed': True,
                'morphology_radius': 2
            }

        # Process images
        if self.n_workers == 1:
            results = self._process_sequential(image_files, scale, analysis_params)
        else:
            results = self._process_parallel(image_files, scale, analysis_params)

        # Save results
        batch_results = self._save_batch_results(
            results, output_dir, save_individual_results, generate_summary
        )

        return batch_results

    def process_file_list(self, file_list: List[str], output_dir: str,
                          scales: Optional[List[float]] = None,
                          analysis_params: Optional[Dict[str, Any]] = None,
                          save_individual_results: bool = True,
                          generate_summary: bool = True) -> Dict[str, Any]:

        if scales is None:
            scales = [1.0] * len(file_list)
        elif len(scales) != len(file_list):
            if len(scales) == 1:
                scales = scales * len(file_list)
            else:
                raise ValueError("Length of scales must match length of file_list or be 1")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set default analysis parameters
        if analysis_params is None:
            analysis_params = {
                'min_area': 50,
                'gaussian_sigma': 1.0,
                'threshold_method': 'otsu',
                'use_watershed': True,
                'morphology_radius': 2
            }

        print(f"Processing {len(file_list)} images")

        # Create job list
        jobs = [(file_path, scale, analysis_params)
                for file_path, scale in zip(file_list, scales)]

        # Process images
        if self.n_workers == 1:
            results = [self._process_single_image(job) for job in jobs]
        else:
            results = self._process_jobs_parallel(jobs)

        # Save results
        batch_results = self._save_batch_results(
            results, output_dir, save_individual_results, generate_summary
        )

        return batch_results

    def compare_conditions(self, condition_dirs: Dict[str, str],
                           output_dir: str, scale: float = 1.0,
                           analysis_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        condition_results = {}

        for condition_name, condition_dir in condition_dirs.items():
            print(f"Processing condition: {condition_name}")

            results = self.process_directory(
                condition_dir,
                os.path.join(output_dir, condition_name),
                scale=scale,
                analysis_params=analysis_params,
                save_individual_results=False,
                generate_summary=False
            )

            condition_results[condition_name] = results

        # Generate comparison report
        comparison_results = self._generate_comparison_report(condition_results, output_dir)

        return comparison_results

    def _find_image_files(self, directory: str, pattern: str) -> List[str]:
        path = Path(directory)

        # Handle multiple patterns
        if isinstance(pattern, str):
            patterns = [pattern]
        else:
            patterns = pattern

        image_files = []
        for pat in patterns:
            image_files.extend(path.glob(pat))

        # Add common image extensions if pattern is generic
        if pattern == "*":
            for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                image_files.extend(path.glob(ext))

        return sorted([str(f) for f in image_files])

    def _process_sequential(self, image_files: List[str], scale: float,
                            analysis_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []

        for i, image_path in enumerate(image_files):
            print(f"Processing {i + 1}/{len(image_files)}: {Path(image_path).name}")

            try:
                result = self._analyze_single_image(image_path, scale, analysis_params)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })

        return results

    def _process_parallel(self, image_files: List[str], scale: float,
                          analysis_params: Dict[str, Any]) -> List[Dict[str, Any]]:

        jobs = [(image_path, scale, analysis_params) for image_path in image_files]
        return self._process_jobs_parallel(jobs)

    def _process_jobs_parallel(self, jobs: List[tuple]) -> List[Dict[str, Any]]:
        results = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._process_single_image, job): job
                for job in jobs
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_job)):
                job = future_to_job[future]
                image_path = job[0]

                print(f"Completed {i + 1}/{len(jobs)}: {Path(image_path).name}")

                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    results.append({
                        'image_path': image_path,
                        'error': str(e),
                        'success': False
                    })

        # Sort results by original order
        job_paths = [job[0] for job in jobs]
        sorted_results = [None] * len(results)

        for result in results:
            if 'image_path' in result:
                try:
                    idx = job_paths.index(result['image_path'])
                    sorted_results[idx] = result
                except ValueError:
                    sorted_results.append(result)

        return [r for r in sorted_results if r is not None]

    def _process_single_image(self, job: tuple) -> Dict[str, Any]:
        image_path, scale, analysis_params = job
        return self._analyze_single_image(image_path, scale, analysis_params)

    def _analyze_single_image(self, image_path: str, scale: float,
                              analysis_params: Dict[str, Any]) -> Dict[str, Any]:

        analyzer = GrainAnalyzer()  # Create new instance for each process

        try:
            # Perform analysis
            results = analyzer.analyze(image_path, scale=scale, **analysis_params)

            # Add metadata
            results.update({
                'image_path': image_path,
                'image_name': Path(image_path).name,
                'scale': scale,
                'analysis_params': analysis_params,
                'success': True,
                'processing_time': datetime.now().isoformat()
            })

            return results

        except Exception as e:
            return {
                'image_path': image_path,
                'image_name': Path(image_path).name,
                'error': str(e),
                'success': False,
                'processing_time': datetime.now().isoformat()
            }

    def _save_batch_results(self, results: List[Dict[str, Any]], output_dir: str,
                            save_individual_results: bool,
                            generate_summary: bool) -> Dict[str, Any]:

        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]

        print(f"Successfully processed: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")

        batch_info = {
            'total_images': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'processing_date': datetime.now().isoformat(),
            'results': successful_results,
            'failed_images': failed_results
        }

        # Save individual results
        if save_individual_results:
            for result in successful_results:
                self._save_individual_result(result, output_dir)

        # Save batch summary
        if generate_summary:
            self._save_batch_summary(batch_info, output_dir)

        # Save failed images log
        if failed_results:
            self._save_failed_log(failed_results, output_dir)

        return batch_info

    def _save_individual_result(self, result: Dict[str, Any], output_dir: str):
        image_name = Path(result['image_name']).stem

        # Save grain metrics CSV
        if 'metrics' in result:
            csv_path = os.path.join(output_dir, f"{image_name}_grains.csv")
            self.exporter.export_csv(result['metrics'], csv_path)

        # Save JSON with all data
        json_path = os.path.join(output_dir, f"{image_name}_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    def _save_batch_summary(self, batch_info: Dict[str, Any], output_dir: str):
        # Create summary statistics
        summary_stats = self._calculate_batch_statistics(batch_info['results'])

        # Save summary CSV
        summary_path = os.path.join(output_dir, "batch_summary.csv")
        self.exporter.export_batch_summary(batch_info['results'], summary_path)

        # Save detailed summary JSON
        detailed_summary = {
            'batch_info': {k: v for k, v in batch_info.items() if k != 'results'},
            'summary_statistics': summary_stats
        }

        json_path = os.path.join(output_dir, "batch_summary.json")
        with open(json_path, 'w') as f:
            json.dump(detailed_summary, f, indent=2, default=str)

        # Generate HTML report
        report_path = os.path.join(output_dir, "batch_report.html")
        self._generate_batch_report(batch_info, summary_stats, report_path)

    def _save_failed_log(self, failed_results: List[Dict[str, Any]], output_dir: str):
        failed_path = os.path.join(output_dir, "failed_images.csv")

        failed_data = []
        for result in failed_results:
            failed_data.append({
                'image_path': result.get('image_path', ''),
                'image_name': result.get('image_name', ''),
                'error': result.get('error', ''),
                'processing_time': result.get('processing_time', '')
            })

        df = pd.DataFrame(failed_data)
        df.to_csv(failed_path, index=False)

    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}

        # Extract key metrics across all images
        all_grain_counts = []
        all_mean_ecds = []
        all_astm_sizes = []

        for result in results:
            if 'statistics' in result:
                stats = result['statistics']

                if 'grain_count' in stats:
                    all_grain_counts.append(stats['grain_count'])

                if 'ecd_statistics' in stats and 'mean' in stats['ecd_statistics']:
                    all_mean_ecds.append(stats['ecd_statistics']['mean'])

                if 'astm_grain_size' in stats and 'grain_size_number' in stats['astm_grain_size']:
                    astm_val = stats['astm_grain_size']['grain_size_number']
                    if astm_val is not None and not pd.isna(astm_val):
                        all_astm_sizes.append(astm_val)

        summary = {
            'total_images_analyzed': len(results),
            'grain_count_statistics': self._calc_stats(all_grain_counts, 'Grain Count'),
            'mean_ecd_statistics': self._calc_stats(all_mean_ecds, 'Mean ECD (μm)'),
            'astm_grain_size_statistics': self._calc_stats(all_astm_sizes, 'ASTM Grain Size')
        }

        return summary

    def _calc_stats(self, values: List[float], name: str) -> Dict[str, Any]:
        if not values:
            return {'name': name, 'count': 0}

        import numpy as np

        return {
            'name': name,
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }

    def _generate_batch_report(self, batch_info: Dict[str, Any],
                               summary_stats: Dict[str, Any],
                               output_path: str):

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
        .summary-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 8px; }}
        .summary-table th {{ background-color: #f2f2f2; }}
        .section {{ margin: 30px 0; }}
        .section h2 {{ color: #2c5aa0; border-bottom: 1px solid #2c5aa0; }}
        .success {{ color: green; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Batch Grain Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>Processing Summary</h2>
        <table class="summary-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Images</td><td>{batch_info['total_images']}</td></tr>
            <tr><td>Successfully Processed</td><td class="success">{batch_info['successful']}</td></tr>
            <tr><td>Failed</td><td class="failed">{batch_info['failed']}</td></tr>
            <tr><td>Success Rate</td><td>{batch_info['success_rate']:.1%}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Cross-Image Statistics</h2>
        {self._create_batch_stats_html(summary_stats)}
    </div>

</body>
</html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

    def _create_batch_stats_html(self, summary_stats: Dict[str, Any]) -> str:
        html = ""

        for stat_name, stat_data in summary_stats.items():
            if isinstance(stat_data, dict) and 'count' in stat_data and stat_data['count'] > 0:
                html += f"<h3>{stat_data['name']}</h3>\n"
                html += '<table class="summary-table">\n'
                html += '<tr><th>Statistic</th><th>Value</th></tr>\n'

                for key, value in stat_data.items():
                    if key != 'name' and isinstance(value, (int, float)):
                        if isinstance(value, float):
                            html += f'<tr><td>{key.title()}</td><td>{value:.3f}</td></tr>\n'
                        else:
                            html += f'<tr><td>{key.title()}</td><td>{value}</td></tr>\n'

                html += '</table>\n'

        return html

    def _generate_comparison_report(self, condition_results: Dict[str, Dict[str, Any]],
                                    output_dir: str) -> Dict[str, Any]:

        # Statistical comparison between conditions
        comparison_stats = {}

        for condition_name, results in condition_results.items():
            if 'results' in results:
                successful_results = [r for r in results['results'] if r.get('success', False)]
                comparison_stats[condition_name] = self._calculate_batch_statistics(successful_results)

        # Save comparison report
        comparison_path = os.path.join(output_dir, "condition_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison_stats, f, indent=2, default=str)

        return {
            'condition_results': condition_results,
            'comparison_statistics': comparison_stats,
            'output_directory': output_dir
        }