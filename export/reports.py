import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import base64
from io import BytesIO


class ReportGenerator:

    def __init__(self):
        self.template_dir = Path(__file__).parent / 'templates'

    def generate_report(self, grain_metrics: Dict[int, Dict[str, Any]],
                        statistics: Dict[str, Any], output_path: str,
                        format_type: str = 'html',
                        include_plots: bool = True) -> str:

        if format_type.lower() == 'html':
            return self._generate_html_report(grain_metrics, statistics, output_path, include_plots)
        elif format_type.lower() == 'markdown':
            return self._generate_markdown_report(grain_metrics, statistics, output_path, include_plots)
        elif format_type.lower() == 'pdf':
            return self._generate_pdf_report(grain_metrics, statistics, output_path, include_plots)
        else:
            raise ValueError(f"Unsupported report format: {format_type}")

    def _generate_html_report(self, grain_metrics: Dict[int, Dict[str, Any]],
                              statistics: Dict[str, Any], output_path: str,
                              include_plots: bool) -> str:

        # Generate plots and get base64 encoded images
        plot_data = {}
        if include_plots:
            plot_data = self._generate_plot_images(grain_metrics, statistics)

        html_content = self._create_html_content(grain_metrics, statistics, plot_data)

        with open(output_path, 'w') as f:
            f.write(html_content)

        return output_path

    def _generate_markdown_report(self, grain_metrics: Dict[int, Dict[str, Any]],
                                  statistics: Dict[str, Any], output_path: str,
                                  include_plots: bool) -> str:

        # Generate plot files if requested
        plot_files = {}
        if include_plots:
            plot_files = self._save_plot_files(grain_metrics, statistics, output_path)

        markdown_content = self._create_markdown_content(grain_metrics, statistics, plot_files)

        with open(output_path, 'w') as f:
            f.write(markdown_content)

        return output_path

    def _generate_pdf_report(self, grain_metrics: Dict[int, Dict[str, Any]],
                             statistics: Dict[str, Any], output_path: str,
                             include_plots: bool) -> str:

        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors

            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                         fontSize=20, spaceAfter=30, alignment=1)
            story.append(Paragraph("Grain Analysis Report", title_style))
            story.append(Spacer(1, 20))

            # Summary section
            story.append(Paragraph("Analysis Summary", styles['Heading2']))
            story.append(Spacer(1, 12))

            summary_data = self._create_summary_table_data(statistics)
            table = Table(summary_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))

            # Build PDF
            doc.build(story)

        except ImportError:
            # Fallback to HTML if reportlab not available
            html_path = output_path.replace('.pdf', '.html')
            return self._generate_html_report(grain_metrics, statistics, html_path, include_plots)

        return output_path

    def _create_html_content(self, grain_metrics: Dict[int, Dict[str, Any]],
                             statistics: Dict[str, Any], plot_data: Dict[str, str]) -> str:

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grain Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 40px;
            color: #333;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .summary-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .plot-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .plot-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section h2 {{
            color: #2c5aa0;
            border-bottom: 1px solid #2c5aa0;
            padding-bottom: 5px;
        }}
        .metadata {{
            background-color: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #2c5aa0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Grain Size Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>Analysis Summary</h2>
        {self._create_html_summary_table(statistics)}
    </div>

    <div class="section">
        <h2>Size Distribution Statistics</h2>
        {self._create_html_statistics_table(statistics)}
    </div>

    {self._create_html_plots_section(plot_data)}

    <div class="section">
        <h2>Grain Details</h2>
        <p>Total number of grains analyzed: <strong>{len(grain_metrics)}</strong></p>
        {self._create_html_grain_table(grain_metrics)}
    </div>

    <div class="metadata">
        <h2>Analysis Metadata</h2>
        <p><strong>Software:</strong> GrainStat v1.0.0</p>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Grains:</strong> {len(grain_metrics)}</p>
    </div>
</body>
</html>
        """

        return html_template

    def _create_markdown_content(self, grain_metrics: Dict[int, Dict[str, Any]],
                                 statistics: Dict[str, Any], plot_files: Dict[str, str]) -> str:

        content = f"""# Grain Size Analysis Report

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Summary

{self._create_markdown_summary_table(statistics)}

## Size Distribution Statistics

{self._create_markdown_statistics_table(statistics)}

{self._create_markdown_plots_section(plot_files)}

## Grain Details

Total number of grains analyzed: **{len(grain_metrics)}**

{self._create_markdown_grain_table(grain_metrics)}

## Analysis Metadata

- **Software:** GrainStat v1.0.0
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Grains:** {len(grain_metrics)}
"""

        return content

    def _create_html_summary_table(self, statistics: Dict[str, Any]) -> str:
        if 'ecd_statistics' not in statistics:
            return "<p>No statistics available</p>"

        ecd_stats = statistics['ecd_statistics']
        astm = statistics.get('astm_grain_size', {})

        rows = [
            ('Total Grains', statistics.get('grain_count', 0)),
            ('Mean ECD (μm)', f"{ecd_stats.get('mean', 0):.2f}"),
            ('Median ECD (μm)', f"{ecd_stats.get('median', 0):.2f}"),
            ('Std Dev ECD (μm)', f"{ecd_stats.get('std', 0):.2f}"),
            ('Min ECD (μm)', f"{ecd_stats.get('min', 0):.2f}"),
            ('Max ECD (μm)', f"{ecd_stats.get('max', 0):.2f}"),
            ('ASTM Grain Size',
             f"{astm.get('grain_size_number', 'N/A'):.1f}" if astm.get('grain_size_number') else 'N/A')
        ]

        table_html = '<table class="summary-table">\n<thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>\n'
        for param, value in rows:
            table_html += f'<tr><td>{param}</td><td>{value}</td></tr>\n'
        table_html += '</tbody></table>'

        return table_html

    def _create_html_statistics_table(self, statistics: Dict[str, Any]) -> str:
        if 'ecd_statistics' not in statistics:
            return "<p>No detailed statistics available</p>"

        ecd_stats = statistics['ecd_statistics']

        rows = [
            ('Count', ecd_stats.get('count', 0)),
            ('Mean (μm)', f"{ecd_stats.get('mean', 0):.3f}"),
            ('Median (μm)', f"{ecd_stats.get('median', 0):.3f}"),
            ('Standard Deviation (μm)', f"{ecd_stats.get('std', 0):.3f}"),
            ('Variance (μm²)', f"{ecd_stats.get('variance', 0):.3f}"),
            ('Skewness', f"{ecd_stats.get('skewness', 0):.3f}"),
            ('Kurtosis', f"{ecd_stats.get('kurtosis', 0):.3f}"),
            ('25th Percentile (μm)', f"{ecd_stats.get('q25', 0):.3f}"),
            ('75th Percentile (μm)', f"{ecd_stats.get('q75', 0):.3f}"),
            ('5th Percentile (μm)', f"{ecd_stats.get('p05', 0):.3f}"),
            ('95th Percentile (μm)', f"{ecd_stats.get('p95', 0):.3f}"),
        ]

        table_html = '<table class="summary-table">\n<thead><tr><th>Statistic</th><th>Value</th></tr></thead>\n<tbody>\n'
        for param, value in rows:
            table_html += f'<tr><td>{param}</td><td>{value}</td></tr>\n'
        table_html += '</tbody></table>'

        return table_html

    def _create_html_plots_section(self, plot_data: Dict[str, str]) -> str:
        if not plot_data:
            return ""

        plots_html = '<div class="section"><h2>Analysis Plots</h2>\n'

        for plot_name, plot_b64 in plot_data.items():
            plots_html += f'''
            <div class="plot-container">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="data:image/png;base64,{plot_b64}" alt="{plot_name}" class="plot-image">
            </div>
            '''

        plots_html += '</div>'
        return plots_html

    def _create_html_grain_table(self, grain_metrics: Dict[int, Dict[str, Any]]) -> str:
        if len(grain_metrics) > 50:
            # Show only first 50 grains for large datasets
            display_metrics = dict(list(grain_metrics.items())[:50])
            note = f"<p><em>Showing first 50 grains out of {len(grain_metrics)} total.</em></p>"
        else:
            display_metrics = grain_metrics
            note = ""

        headers = ['Grain ID', 'ECD (μm)', 'Area (μm²)', 'Aspect Ratio', 'Shape Factor', 'Eccentricity']

        table_html = note + '<table class="summary-table">\n<thead><tr>'
        for header in headers:
            table_html += f'<th>{header}</th>'
        table_html += '</tr></thead>\n<tbody>\n'

        for grain_id, metrics in display_metrics.items():
            table_html += f'<tr>'
            table_html += f'<td>{grain_id}</td>'
            table_html += f'<td>{metrics.get("ecd_um", 0):.2f}</td>'
            table_html += f'<td>{metrics.get("area_um2", 0):.2f}</td>'
            table_html += f'<td>{metrics.get("aspect_ratio", 0):.2f}</td>'
            table_html += f'<td>{metrics.get("shape_factor", 0):.3f}</td>'
            table_html += f'<td>{metrics.get("eccentricity", 0):.3f}</td>'
            table_html += '</tr>\n'

        table_html += '</tbody></table>'
        return table_html

    def _create_markdown_summary_table(self, statistics: Dict[str, Any]) -> str:
        if 'ecd_statistics' not in statistics:
            return "No statistics available"

        ecd_stats = statistics['ecd_statistics']
        astm = statistics.get('astm_grain_size', {})

        table = """| Parameter | Value |
|-----------|-------|
"""

        rows = [
            ('Total Grains', statistics.get('grain_count', 0)),
            ('Mean ECD (μm)', f"{ecd_stats.get('mean', 0):.2f}"),
            ('Median ECD (μm)', f"{ecd_stats.get('median', 0):.2f}"),
            ('Std Dev ECD (μm)', f"{ecd_stats.get('std', 0):.2f}"),
            ('Min ECD (μm)', f"{ecd_stats.get('min', 0):.2f}"),
            ('Max ECD (μm)', f"{ecd_stats.get('max', 0):.2f}"),
            ('ASTM Grain Size',
             f"{astm.get('grain_size_number', 'N/A'):.1f}" if astm.get('grain_size_number') else 'N/A')
        ]

        for param, value in rows:
            table += f"| {param} | {value} |\n"

        return table

    def _create_markdown_statistics_table(self, statistics: Dict[str, Any]) -> str:
        if 'ecd_statistics' not in statistics:
            return "No detailed statistics available"

        ecd_stats = statistics['ecd_statistics']

        table = """| Statistic | Value |
|-----------|-------|
"""

        rows = [
            ('Count', ecd_stats.get('count', 0)),
            ('Mean (μm)', f"{ecd_stats.get('mean', 0):.3f}"),
            ('Median (μm)', f"{ecd_stats.get('median', 0):.3f}"),
            ('Standard Deviation (μm)', f"{ecd_stats.get('std', 0):.3f}"),
            ('Variance (μm²)', f"{ecd_stats.get('variance', 0):.3f}"),
            ('Skewness', f"{ecd_stats.get('skewness', 0):.3f}"),
            ('Kurtosis', f"{ecd_stats.get('kurtosis', 0):.3f}"),
            ('25th Percentile (μm)', f"{ecd_stats.get('q25', 0):.3f}"),
            ('75th Percentile (μm)', f"{ecd_stats.get('q75', 0):.3f}"),
        ]

        for param, value in rows:
            table += f"| {param} | {value} |\n"

        return table

    def _create_markdown_plots_section(self, plot_files: Dict[str, str]) -> str:
        if not plot_files:
            return ""

        plots_md = "\n## Analysis Plots\n\n"

        for plot_name, plot_file in plot_files.items():
            plots_md += f"### {plot_name.replace('_', ' ').title()}\n\n"
            plots_md += f"![{plot_name}]({plot_file})\n\n"

        return plots_md

    def _create_markdown_grain_table(self, grain_metrics: Dict[int, Dict[str, Any]]) -> str:
        if len(grain_metrics) > 50:
            display_metrics = dict(list(grain_metrics.items())[:50])
            note = f"*Showing first 50 grains out of {len(grain_metrics)} total.*\n\n"
        else:
            display_metrics = grain_metrics
            note = ""

        table = note + """| Grain ID | ECD (μm) | Area (μm²) | Aspect Ratio | Shape Factor | Eccentricity |
|----------|----------|------------|--------------|--------------|--------------|
"""

        for grain_id, metrics in display_metrics.items():
            table += f"| {grain_id} | {metrics.get('ecd_um', 0):.2f} | {metrics.get('area_um2', 0):.2f} | "
            table += f"{metrics.get('aspect_ratio', 0):.2f} | {metrics.get('shape_factor', 0):.3f} | "
            table += f"{metrics.get('eccentricity', 0):.3f} |\n"

        return table

    def _generate_plot_images(self, grain_metrics: Dict[int, Dict[str, Any]],
                              statistics: Dict[str, Any]) -> Dict[str, str]:
        from ..visualization.plots import PlotGenerator

        plotter = PlotGenerator()
        plot_data = {}

        try:
            # Generate histogram
            fig = plotter.plot_histogram(grain_metrics, bins=30)
            plot_data['size_histogram'] = self._fig_to_base64(fig)
            plt.close(fig)

            # Generate cumulative distribution
            fig = plotter.plot_cumulative_distribution(grain_metrics)
            plot_data['cumulative_distribution'] = self._fig_to_base64(fig)
            plt.close(fig)

            # Generate shape analysis
            fig = plotter.plot_shape_analysis(grain_metrics)
            plot_data['shape_analysis'] = self._fig_to_base64(fig)
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Could not generate plots for report: {e}")

        return plot_data

    def _save_plot_files(self, grain_metrics: Dict[int, Dict[str, Any]],
                         statistics: Dict[str, Any], output_path: str) -> Dict[str, str]:
        from ..visualization.plots import PlotGenerator

        plotter = PlotGenerator()
        plot_files = {}

        base_path = Path(output_path).parent
        base_name = Path(output_path).stem

        try:
            # Generate and save histogram
            fig = plotter.plot_histogram(grain_metrics, bins=30)
            hist_path = base_path / f"{base_name}_histogram.png"
            fig.savefig(hist_path, dpi=300, bbox_inches='tight')
            plot_files['size_histogram'] = str(hist_path)
            plt.close(fig)

            # Generate and save cumulative distribution
            fig = plotter.plot_cumulative_distribution(grain_metrics)
            cdf_path = base_path / f"{base_name}_cumulative.png"
            fig.savefig(cdf_path, dpi=300, bbox_inches='tight')
            plot_files['cumulative_distribution'] = str(cdf_path)
            plt.close(fig)

            # Generate and save shape analysis
            fig = plotter.plot_shape_analysis(grain_metrics)
            shape_path = base_path / f"{base_name}_shape.png"
            fig.savefig(shape_path, dpi=300, bbox_inches='tight')
            plot_files['shape_analysis'] = str(shape_path)
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Could not generate plot files: {e}")

        return plot_files

    def _fig_to_base64(self, fig) -> str:
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64

    def _create_summary_table_data(self, statistics: Dict[str, Any]) -> list:
        if 'ecd_statistics' not in statistics:
            return [['Parameter', 'Value'], ['No data', 'available']]

        ecd_stats = statistics['ecd_statistics']
        astm = statistics.get('astm_grain_size', {})

        return [
            ['Parameter', 'Value'],
            ['Total Grains', str(statistics.get('grain_count', 0))],
            ['Mean ECD (μm)', f"{ecd_stats.get('mean', 0):.2f}"],
            ['Median ECD (μm)', f"{ecd_stats.get('median', 0):.2f}"],
            ['Std Dev ECD (μm)', f"{ecd_stats.get('std', 0):.2f}"],
            ['ASTM Grain Size',
             f"{astm.get('grain_size_number', 'N/A'):.1f}" if astm.get('grain_size_number') else 'N/A']
        ]