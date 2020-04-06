import sys
import csv
import codecs
from datetime import date
from raw_corpus import corpus_builder

def main() :
    stats = dict()
    cat_stats = dict()

    # Instantiate a Raw corpus builder 
    cb = corpus_builder('./corpus')

    # Process the events csv file
    count = 0
    attributes = ['name', 'start_date', 'end_date', 'price', 'address', 'zipcode', 'city', 'tag', 'media', 'email', 'telephone', 'description', 'class', ]
    with codecs.open('events.csv', 'r', "utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        headers=reader.fieldnames
        for row in reader :
            category=row['evt_category']
            evt_class=row['class']
            #if evt_class not in ['artistic-event'] :
            #    continue
            if len(row['description']) == 0 :
                continue
            #print('{:32.32s} {:32.32s} {:32.32s} {:s} {:16.16s} -> [{:16.16s}]'.format(row['name'],row['venue_name'], row['address'], row['zipcode'], row['city'], evt_class))
            count += 1

            row['slug']=row['source_url']
            row['description']='<p>'+row['description']+'</p>'
            cb.process(row, evt_class)
            # Statistics 
            if row['venue_name'] in stats :
                stats[row['venue_name']] += 1
            else :
                stats[row['venue_name']] = 1
            if category not in cat_stats :
                cat_stats[category] = [row['name'],]
            else :
                cat_stats[category].append(row['name'])
    print('{:d} events'.format(count))
    for venue in stats :
        print('{:s} : {:d} events'.format(venue, stats[venue]))

    for category in cat_stats :
        print('== '+category+' ==')
        for evt in cat_stats[category] :
            print('\t'+evt)
if __name__ == "__main__":
    main()
