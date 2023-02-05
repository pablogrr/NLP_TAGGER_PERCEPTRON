from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures

# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

    def add_emission_features(self, sequence, pos, y, features):

        #Based on the current label
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        # Suffixes
        max_suffix = 3
        for i in range(max_suffix):
            if len(word) > i+1:
                suffix = word[-(i+1):]
                # Generate feature name.
                feat_name = "suffix:%s::%s" % (suffix, y_name)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        # Prefixes
        max_prefix = 3
        for i in range(max_prefix):
            if len(word) > i+1:
                prefix = word[:(i+1)]
                # Generate feature name.
                feat_name = "prefix:%s::%s" % (prefix, y_name)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        #Lists of common words in some categories

        honorifics = ["Mr.", "Ms.", "Miss.", "Mrs.", "President", "Vice", "Prime", "Major", "Liutenant", "General", "Sister", "Father",
                      "Prophet", "Pope", "Senator", "Deputy", "Chairman", "Dr.", "Admiral", "King", "Queen"]
        
        if word in honorifics:
            feat_name = "honor::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        org_names = ["International", "National", "Union", "United", "Association", "Associated", "Office", "Committee",
                     "Organization", "Ministry", "Bank", "University", "Institute", "Department", "Agency", "Commission"]

        if word in org_names:
            feat_name = "org_names::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)
                
        geo_indicators = ["Middle", "North", "South", "East", "West", "Northern", "Southern", "Eastern", "Western",
                               "New"]

        if word in geo_indicators:
            feat_name = "geo_indicators::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)        

        geo_suf = ["State", "States", "City", "Park"]

        if word in geo_suf:
            feat_name = "geo_suf::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        countries=['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola',
                   'Antigua', 'Argentina', 'Armenia', 'Australia', 'Austria',
                   'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados',
                   'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia',
                   'Bosnia', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria',
                   'Burkina', 'Burundi', 'Cambodia', 'Cameroon', 'Canada',
                   'Cape', 'Central', 'Chad', 'Chile', 'China', 'Colombia',
                   'Comoros', 'Congo', 'Costa', 'Croatia', 'Cuba', 'Cyprus',
                   'Czech', 'Denmark', 'Djibouti', 'Dominica', 'Dominican',
                   'East', 'Ecuador', 'Egypt', 'El', 'Equatorial', 'Eritrea',
                   'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France',
                   'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece',
                   'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana',
                   'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India',
                   'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
                   'Ivory', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya',
                   'Kiribati', 'Korea', 'Kosovo', 'Kuwait', 'Kyrgyzstan',
                   'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya',
                   'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia',
                   'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali',
                   'Malta', 'Marshall', 'Mauritania', 'Mauritius', 'Mexico',
                   'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro',
                   'Morocco', 'Mozambique', 'Myanmar,', 'Namibia', 'Nauru',
                   'Nepal', 'Netherlands', 'New', 'Nicaragua', 'Niger',
                   'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama',
                   'Papua', 'Paraguay', 'Peru', 'Philippines', 'Poland',
                   'Portugal', 'Qatar', 'Romania', 'Russian', 'Rwanda',
                   'Saint', 'Samoa', 'San', 'Sao', 'Saudi', 'Senegal',
                   'Serbia', 'Seychelles', 'Sierra', 'Singapore', 'Slovakia',
                   'Slovenia', 'Solomon', 'Somalia', 'South', 'Spain', 'Sri',
                   'St', 'Sudan', 'Suriname', 'Swaziland', 'Sweden',
                   'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania',
                   'Thailand', 'Togo', 'Tonga', 'Trinidad', 'Tunisia', 'Turkey',
                   'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United',
                   'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican', 'Venezuela',
                   'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']        

        if word in countries:
            feat_name = "country::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        if word[0].isupper():
            feat_name = "firstupper::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        if word.isdigit():
            feat_name = "digit::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        if word.endswith("th"):  
            feat_name = "order::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        if ((not str.isalnum(word)) and len(word) == 1):
            feat_name = "symbol:%s::%s" % (word, y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        if str.istitle(word):
            feat_name = "title::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        #Based on the previous label
        if pos!=0:    
            x = sequence.x[pos-1]

            # Get word name from ID.
            if isinstance(x, str):
                x_name = x
            else:
                x_name = self.dataset.x_dict.get_label_name(x)

            word = str(x_name)
            # Generate feature name.
            feat_name = "id_prev:%s::%s" % (word, y_name)
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        #Based on the next label
        if pos!=len(sequence.x)-1:    
            x = sequence.x[pos+1]

            # Get word name from ID.
            if isinstance(x, str):
                x_name = x
            else:
                x_name = self.dataset.x_dict.get_label_name(x)

            word = str(x_name)
            # Generate feature name.
            feat_name = "id_next:%s::%s" % (word, y_name)
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id) 

        return features







































class ExtendedUnicodeFeatures(UnicodeFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = y

        # Get word name from ID.
        x_name = x

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        feat_name = str(feat_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        if str.istitle(word):
            # Generate feature name.
            feat_name = "uppercased::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if str.isdigit(word):
            # Generate feature name.
            feat_name = "number::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if str.find(word, "-") != -1:
            # Generate feature name.
            feat_name = "hyphen::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Suffixes
        max_suffix = 3
        for i in range(max_suffix):
            if len(word) > i+1:
                suffix = word[-(i+1):]
                # Generate feature name.
                feat_name = "suffix:%s::%s" % (suffix, y_name)
                feat_name = str(feat_name)

                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        # Prefixes
        max_prefix = 3
        for i in range(max_prefix):
            if len(word) > i+1:
                prefix = word[:i+1]
                # Generate feature name.
                feat_name = "prefix:%s::%s" % (prefix, y_name)
                feat_name = str(feat_name)

                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        return features
