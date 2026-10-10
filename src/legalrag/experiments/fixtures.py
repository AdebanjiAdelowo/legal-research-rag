"""Hand-written questions for scope detection and abstention.

``OUT_OF_SCOPE`` asks about law the corpus does not hold: state law, the law of
other countries, or nothing legal at all. A system limited to U.S. federal opinions
should decline each one. ``IN_SCOPE`` asks federal questions phrased the way a
researcher would, several of which mention a state or a foreign party without being
about that law, to measure false alarms.

These are fixed lists, written before any experiment was run. They are small and
are not a sample of real user questions.
"""

OUT_OF_SCOPE = {
    "state": [
        "What does the California Supreme Court say about unconscionability in consumer contracts?",
        "Is a non-compete agreement enforceable under Texas law?",
        "What is the statute of limitations for medical malpractice under New York state law?",
        "How does the Florida Supreme Court define premises liability for invitees?",
        "What are the elements of adverse possession under Ohio law?",
        "Does Illinois law recognise a claim for negligent infliction of emotional distress?",
        "How is community property divided on divorce under Washington state law?",
        "What does the New Jersey Supreme Court require for a valid prenuptial agreement?",
        "Under Pennsylvania law, when is a landlord liable for criminal acts of third parties?",
        "What is the standard for piercing the corporate veil in a Delaware state court?",
        "How does the Michigan Court of Appeals treat implied warranties of habitability?",
        "What notice must an employer give under the Massachusetts statute on wage payment?",
        "When does the Georgia Supreme Court allow punitive damages in a tort action?",
        "What are the state-law requirements for a holographic will in Arizona law?",
        "How does the Oregon Court of Appeals apply the economic loss rule?",
    ],
    "foreign": [
        "What is the right to erasure under the GDPR?",
        "How do courts in England decide when a duty of care arises in negligence?",
        "What did the House of Lords hold about psychiatric injury to secondary victims?",
        "How does Canadian law treat unreasonable search and seizure under the Charter?",
        "What is the test for unfair dismissal in Australia?",
        "How does German law regulate standard contract terms?",
        "What does the Court of Justice of the European Union require for a preliminary reference?",
        "What are directors' duties under Nigerian company law?",
        "How does Indian law define the basic structure doctrine?",
        "What is the limitation period for contract claims under French law?",
        "How do courts in Singapore approach contractual penalty clauses?",
        "What does South African law say about the right to housing?",
        "How does Japanese law treat lifetime employment contracts?",
        "What is judicial review of administrative action in Ireland based on?",
        "How does Italian law protect geographical indications?",
    ],
    "non_legal": [
        "What is the best way to prepare sourdough starter at home?",
        "How does a transformer neural network compute attention?",
        "Which planets in the solar system have rings?",
        "What were the main causes of the fall of the Western Roman Empire?",
        "How do I change a flat tyre on a bicycle?",
        "What is the boiling point of ethanol at sea level?",
        "Who won the football world cup in 1998?",
        "How should I structure a weekly marathon training plan?",
        "What is the difference between a virus and a bacterium?",
        "How do tides form in the ocean?",
    ],
}

IN_SCOPE = [
    "What is the standard for granting summary judgment in federal court?",
    "When does a government official lose qualified immunity?",
    "What must a plaintiff show to have Article III standing?",
    "What are the requirements for a preliminary injunction in the Ninth Circuit?",
    "How do federal courts review an agency's interpretation of a statute it administers?",
    "When is a complaint sufficient to survive a motion to dismiss for failure to state a claim?",
    "What is the standard for ineffective assistance of counsel under the Sixth Amendment?",
    "When may a federal court grant habeas relief to a state prisoner?",
    "What does the Fourth Amendment require for a warrantless search of a car in New York?",
    "How is personal jurisdiction established over a corporation headquartered in Germany?",
    "What is the burden-shifting framework for employment discrimination under Title VII?",
    "When is a class action certified under Rule 23?",
    "What is the standard of review for a district court's evidentiary rulings?",
    "When does the Eleventh Amendment bar a suit against a state official?",
    "How do courts decide whether expert testimony is admissible?",
    "What is required to prove municipal liability under section 1983?",
    "When is an arbitration agreement enforceable under the Federal Arbitration Act?",
    "What is the test for a regulatory taking under the Fifth Amendment?",
    "When does a federal court have supplemental jurisdiction over related claims?",
    "What deference do courts give the Board of Immigration Appeals?",
    "What is the harmless error standard on direct appeal in a criminal case?",
    "How do courts apply strict scrutiny to content-based restrictions on speech?",
    "What is the standard for a certificate of appealability?",
    "When is a Texas police officer entitled to qualified immunity in a federal civil rights suit?",
    "How does the Second Circuit define irreparable harm before 2008?",
    "What must be shown to establish a prima facie case of retaliation?",
    "When does res judicata bar a later federal action?",
    "What is the plain error standard under Rule 52(b)?",
    "How do federal courts treat a claim of prosecutorial misconduct in closing argument?",
    "What is the standard for pleading fraud with particularity under Rule 9(b)?",
]
