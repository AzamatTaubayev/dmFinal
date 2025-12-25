import requests
import time
import json
import pandas as pd
from datetime import datetime
import os


class HHKzDataCollector:
    """Collector for hh.kz job vacancy data"""

    def __init__(self):
        self.base_url = "https://api.hh.ru"
        self.vacancies_endpoint = "/vacancies"
        self.vacancy_detail_endpoint = "/vacancies/{}"
        self.delay = 2  # 2 seconds between requests (be respectful)

    def search_vacancies(self, search_params):
        """Search for vacancies with given parameters"""
        url = self.base_url + self.vacancies_endpoint

        try:
            response = requests.get(url, params=search_params)
            time.sleep(self.delay)  # Rate limiting

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            print(f"Request error: {e}")
            return None

    def get_vacancy_details(self, vacancy_id):
        """Get detailed information for a specific vacancy"""
        url = self.base_url + self.vacancy_detail_endpoint.format(vacancy_id)

        try:
            response = requests.get(url)
            time.sleep(self.delay)  # Rate limiting

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting vacancy {vacancy_id}: {response.status_code}")
                return None

        except Exception as e:
            print(f"Request error for vacancy {vacancy_id}: {e}")
            return None

    def collect_it_vacancies(self, max_pages=30):
        """
        Collect IT vacancies from Kazakhstan
        max_pages: Number of pages to collect (100 items per page max)
        """

        # Professional roles for IT (from hh.ru API documentation)
        it_roles = [
            96,  # Programmer, developer
            104,  # System administrator
            113,  # Analyst
            124,  # Tester, QA
            125,  # DevOps
            126,  # Data Scientist
            # Add more if needed
        ]

        all_vacancy_ids = []

        search_params = {
            'area': 40,  # Kazakhstan
            'professional_role': it_roles,
            'per_page': 100,  # Max per page
            'page': 0
        }

        print("Starting vacancy collection...")

        for page in range(max_pages):
            search_params['page'] = page

            print(f"\nFetching page {page + 1}/{max_pages}...")
            result = self.search_vacancies(search_params)

            if not result:
                print("No results, stopping...")
                break

            items = result.get('items', [])
            if not items:
                print("No more items, stopping...")
                break

            # Collect vacancy IDs
            for item in items:
                all_vacancy_ids.append(item['id'])

            print(f"Collected {len(items)} vacancy IDs (Total: {len(all_vacancy_ids)})")

            # Check if we've reached the last page
            if page >= result.get('pages', 0) - 1:
                print("Reached last page")
                break

        print(f"\n✓ Total vacancy IDs collected: {len(all_vacancy_ids)}")
        return list(set(all_vacancy_ids))  # Remove duplicates

    def collect_detailed_data(self, vacancy_ids, output_file='vacancies_detailed.json'):
        """
        Collect detailed information for each vacancy
        """
        detailed_vacancies = []
        total = len(vacancy_ids)

        print(f"\nCollecting detailed data for {total} vacancies...")

        for idx, vacancy_id in enumerate(vacancy_ids, 1):
            print(f"Processing {idx}/{total}: Vacancy ID {vacancy_id}")

            details = self.get_vacancy_details(vacancy_id)

            if details:
                detailed_vacancies.append(details)

                # Save periodically (every 50 vacancies)
                if idx % 50 == 0:
                    self.save_data(detailed_vacancies, output_file)
                    print(f"  → Saved checkpoint at {idx} vacancies")

        # Final save
        self.save_data(detailed_vacancies, output_file)
        print(f"\n✓ Collected {len(detailed_vacancies)} detailed vacancies")
        return detailed_vacancies

    def save_data(self, data, filename):
        """Save data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def convert_to_csv(self, json_file, csv_file='vacancies.csv'):
        """Convert JSON data to CSV with key fields"""

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract key fields
        rows = []
        for vacancy in data:
            row = {
                'vacancy_id': vacancy.get('id'),
                'name': vacancy.get('name'),
                'area': vacancy.get('area', {}).get('name'),
                'salary_from': vacancy.get('salary', {}).get('from') if vacancy.get('salary') else None,
                'salary_to': vacancy.get('salary', {}).get('to') if vacancy.get('salary') else None,
                'salary_currency': vacancy.get('salary', {}).get('currency') if vacancy.get('salary') else None,
                'experience': vacancy.get('experience', {}).get('name'),
                'schedule': vacancy.get('schedule', {}).get('name'),
                'employment': vacancy.get('employment', {}).get('name'),
                'description': vacancy.get('description'),
                'key_skills': ', '.join([skill['name'] for skill in vacancy.get('key_skills', [])]),
                'employer_name': vacancy.get('employer', {}).get('name'),
                'published_at': vacancy.get('published_at'),
                'url': vacancy.get('alternate_url')
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\n✓ Saved CSV with {len(df)} rows to {csv_file}")
        return df


# Main execution
if __name__ == "__main__":
    collector = HHKzDataCollector()

    # Step 1: Collect vacancy IDs (quick)
    print("=" * 60)
    print("STEP 1: Collecting vacancy IDs")
    print("=" * 60)
    vacancy_ids = collector.collect_it_vacancies(max_pages=30)

    # Save IDs
    with open('vacancy_ids.json', 'w') as f:
        json.dump(vacancy_ids, f)

    # Step 2: Collect detailed data (slow - will take hours)
    print("\n" + "=" * 60)
    print("STEP 2: Collecting detailed vacancy data")
    print("=" * 60)
    print(f"This will take approximately {len(vacancy_ids) * 2 / 60:.1f} minutes")

    proceed = input("\nProceed with detailed collection? (yes/no): ")

    if proceed.lower() == 'yes':
        detailed_data = collector.collect_detailed_data(
            vacancy_ids,
            output_file='vacancies_detailed.json'
        )

        # Step 3: Convert to CSV
        print("\n" + "=" * 60)
        print("STEP 3: Converting to CSV")
        print("=" * 60)
        df = collector.convert_to_csv('vacancies_detailed.json')

        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE!")
        print("=" * 60)
        print(f"Files created:")
        print(f"  - vacancy_ids.json ({len(vacancy_ids)} IDs)")
        print(f"  - vacancies_detailed.json (full data)")
        print(f"  - vacancies.csv (structured data)")
    else:
        print("\nCollection cancelled. Run again when ready.")